"""S9 saved-table analysis. No dataset/model imports; execute real tables in Slurm."""
import argparse
import csv
import hashlib
import itertools
import contextlib
import io
import os
import sys
import scipy
import json
import time
from pathlib import Path
import numpy as np
from scipy.optimize import linprog

ACTIONS=[f'p{p}-r{r}' for p in (24,32,49) for r in (4,8,16)]
LOOK=np.array([p-24 for p in (24,32,49) for _ in (4,8,16)],dtype=np.float64)
FEATURES=['left_rate', 'right_rate', 'left_zero_fraction', 'right_zero_fraction', 'left_count_gradient', 'right_count_gradient', 'disparity_mean', 'disparity_std', 'increment3to4_absmean', 'increment2to3_absmean', 'increment3to4_gt1_fraction', 'fine_entropy_mean', 'fine_entropy_std', 'fine_zmargin_mean']
GRID=np.array([0,.01,.03,.1,.3,1,3,10,30,100],dtype=np.float64)
SOLVER={'method':'highs','options':{'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9}}

def ridge_fit(x,y):
    x,y=np.asarray(x,dtype=np.float64),np.asarray(y,dtype=np.float64)
    mean=x.mean(0);sd=np.maximum(x.std(0),1e-6)
    z=np.clip((x-mean)/sd,-5,5);design=np.column_stack([np.ones(len(x)),z,z*z])
    penalty=np.eye(design.shape[1])*.01;penalty[0,0]=0
    beta=np.linalg.solve(design.T@design/len(x)+penalty,design.T@y/len(x))
    return {'mean':mean,'sd':sd,'beta':beta}

def ridge_predict(fit,x):
    z=np.clip((np.asarray(x,dtype=np.float64)-fit['mean'])/fit['sd'],-5,5)
    return np.column_stack([np.ones(len(z)),z,z*z])@fit['beta']

def matrix(rows,field):
    if field=='cost_direct':return np.array([[r['actions'][a]['gpu_ms_direct'] for a in ACTIONS] for r in rows],dtype=np.float64)
    return np.array([[r['actions'][a][field] for a in ACTIONS] for r in rows],dtype=np.float64)

def costs(rows,key):
    if key=='direct':return matrix(rows,'cost_direct')
    return np.array([[r['policy_gpu_ms'][key].get(a,np.nan) for a in ACTIONS] for r in rows],dtype=np.float64)

def onehot(indices):return np.eye(9)[indices]

def reduction(rows,weights,gpu):
    """Expected frame metrics first, then equal frame->scene; pooled bad3 separate."""
    w=np.asarray(weights); assert w.shape==(len(rows),9)
    assert np.isfinite(w).all() and (w>=-1e-10).all() and np.allclose(w.sum(1),1,atol=1e-9,rtol=0)
    def expected(m):return np.sum(np.where(w>0,w*m,0),axis=1)
    e=expected(matrix(rows,'epe'));b=expected(matrix(rows,'bad3_percent'));g=expected(gpu)
    l=w@LOOK; scene={}
    for sid in sorted({r['source_id'] for r in rows}):
        ix=[i for i,r in enumerate(rows) if r['source_id']==sid]
        scene[sid]={'epe':float(e[ix].mean()),'bad3_percent':float(b[ix].mean()),'gpu_ms':float(g[ix].mean()),'lookahead_bins':float(l[ix].mean()),'frames':len(ix)}
    vals=list(scene.values());vc=np.array([r['valid_pixels'] for r in rows],dtype=np.float64)
    return {'epe':float(np.mean([v['epe'] for v in vals])),'bad3_percent':float(np.mean([v['bad3_percent'] for v in vals])),'pooled_bad3_percent':float(100*np.sum(expected(matrix(rows,'bad3_count')))/vc.sum()),'worst5_epe':float(np.mean(sorted([v['epe'] for v in vals])[-5:])),'gpu_ms':float(np.mean([v['gpu_ms'] for v in vals])),'lookahead_bins':float(np.mean([v['lookahead_bins'] for v in vals])),'scenes':scene,'action_frequency':dict(zip(ACTIONS,map(float,w.mean(0)))),'frame_weights':w.tolist()}

def choose_penalties(pred,actual,cost,lookcap,computecap,T,E0,allowed):
    """Selection uses calibration only, including actual EPE and mean cost table."""
    mean=cost[:,allowed].mean(0);best=None
    for lam,mu in itertools.product(GRID*E0,repeat=2):
        score=pred[:,allowed]-lam*LOOK[allowed]/25-mu*mean/T
        ix=np.asarray(allowed)[np.argmax(score,axis=1)]
        c=float(cost[np.arange(len(ix)),ix].mean());l=float(LOOK[ix].mean())
        if l>lookcap or c>.95*computecap:continue
        e=float(actual[np.arange(len(ix)),ix].mean());key=(e,c,l,float(lam),float(mu))
        if best is None or key<best[0]:best=(key,{'lambda':float(lam),'mu':float(mu),'allowed':list(allowed),'calibration_mean_action_cost':mean.tolist(),'calibration_epe':e,'calibration_gpu_ms':c,'calibration_lookahead_bins':l})
    return None if best is None else best[1]

def selected_weights(pred,fit,T):
    a=np.array(fit['allowed']);score=pred[:,a]-fit['lambda']*LOOK[a]/25-fit['mu']*np.array(fit['calibration_mean_action_cost'])/T
    return onehot(a[np.argmax(score,axis=1)])

def lp_allocation(errors,costs_,lookcap,computecap,allowed,shared=False,tie=False):
    """Primal LP; no changes to caps. Optional fixed-mixture lexicographic ties."""
    n=len(errors); a=np.array(allowed);m=len(a)
    if shared:
        obj=errors[:,a].mean(0);cg=costs_[:,a].mean(0);cl=LOOK[a]
        eq=np.ones((1,m));beq=np.ones(1)
    else:
        obj=errors[:,a].ravel()/n;cg=costs_[:,a].ravel()/n;cl=np.tile(LOOK[a],n)/n
        from scipy.sparse import kron,eye
        eq=kron(eye(n),np.ones((1,m))).tocsr();beq=np.ones(n)
    ub=np.vstack([cg,cl]);bu=np.array([computecap,lookcap])
    sol=linprog(obj,A_ub=ub,b_ub=bu,A_eq=eq,b_eq=beq,bounds=(0,1),**SOLVER)
    if not sol.success:return {'feasible':False,'solver_status':sol.status,'message':sol.message}
    if tie:
        # Objective, compute and lookahead are successively fixed at their solved values.
        # Equalities introduce no scientific cap slack; residuals are reported explicitly.
        ee=np.asarray(eq);bb=beq.copy()
        for fix,next_obj in [(obj,cg),(cg,cl)]+[(cl if j==0 else np.eye(m)[j-1],np.eye(m)[j]) for j in range(m)]:
            ee=np.vstack([ee,fix]);bb=np.append(bb,float(fix@sol.x))
            nxt=linprog(next_obj,A_ub=ub,b_ub=bu,A_eq=ee,b_eq=bb,bounds=(0,1),**SOLVER)
            if not nxt.success:raise RuntimeError('LP lexicographic tie solve failed: '+nxt.message)
            sol=nxt
    ww=np.zeros((n,9));ww[:,a]=sol.x[None,:] if shared else sol.x.reshape(n,m)
    return {'feasible':True,'weights':ww,'objective':float(np.sum(ww*errors)/n),'gpu_ms':float(np.sum(np.where(ww>0,ww*costs_,0))/n),'lookahead_bins':float(np.sum(ww*LOOK)/n),'max_probability_sum_error':float(abs(ww.sum(1)-1).max()),'minimum_probability':float(ww.min()),'solver_status':sol.status,'solver':SOLVER,'compute_cap_residual':float(computecap-np.sum(np.where(ww>0,ww*costs_,0))/n),'lookahead_cap_residual':float(lookcap-np.sum(ww*LOOK)/n)}

def random_allocation(cost,lookcap,computecap):
    stop=cost[:,0].mean();uniform=cost.mean();dl=LOOK.mean();alpha=1.
    if stop>computecap:return None
    if uniform>stop:alpha=min(alpha,(computecap-stop)/(uniform-stop))
    alpha=min(alpha,lookcap/dl)
    p=np.full(9,alpha/9);p[0]+=1-alpha
    return np.repeat(p[None,:],len(cost),axis=0)

def fit_family(cal,dev,key,features,allowed,start,T,L,C,name):
    ce,de=matrix(cal,'epe'),matrix(dev,'epe');cc,dc=costs(cal,key),costs(dev,key)
    y=ce[:,start,None]-ce;y[:,start]=0
    f=ridge_fit(features[0],y);cp,dp=ridge_predict(f,features[0]),ridge_predict(f,features[1]);cp[:,start]=0;dp[:,start]=0
    pick=choose_penalties(cp,ce,cc,L,C,T,float(ce[:,start].mean()),allowed)
    if pick is None:return {'name':name,'feasible':False}
    w=selected_weights(dp,pick,T);res=reduction(dev,w,dc)
    return {'name':name,'feasible':True,'calibration_selection':pick,'ridge':f,'result':res}

def feature_matrix(rows,p):return np.array([r['features'][f'p{p}'] for r in rows],dtype=np.float64)

def scene_wins(j,c):return sum(j['scenes'][s]['epe']<v['epe'] for s,v in c['scenes'].items())

def compare(j,c):
    margin=max(.10,.05*c['epe'])
    return {'epe_margin_required':margin,'epe_improvement':c['epe']-j['epe'],'scene_wins':scene_wins(j,c),'pass':bool(c['epe']-j['epe']>=margin and j['bad3_percent']<=c['bad3_percent']+.5 and j['worst5_epe']<=1.05*c['worst5_epe'] and scene_wins(j,c)>=20)}

def bound_cohort(rows,group):
    path=Path(__file__).resolve().parents[1]/'configs/stage9-split-rows.json'
    assert hashlib.sha256(path.read_bytes()).hexdigest()=='ce1a878a7687ec374f9d88a23701ca196902c3f9ae8e955be2435b443491a777'
    binding=json.loads(path.read_text())
    expected={r['sample_id']:r['source_id'] for r in binding['rows'] if r['group']==group}
    actual={r['sample_id']:r['source_id'] for r in rows}
    assert len(actual)==len(rows) and actual==expected, 'exact bound split/cohort mismatch'

def fit_calibration_policies(cal,T):
    """Train/calibration-only API; no Dev/F rows, returned packet is frozen."""
    assert len(cal)==160 and all(r['arm']=='V' and r['split']=='calibration' for r in cal)
    bound_cohort(cal,'policy_calibration')
    ce=matrix(cal,'epe');packet={'T_gpu_ms':float(T),'feature_names':FEATURES,'cells':[]}
    for L,mult in itertools.product((8,16),(1.,1.5)):
        C=mult*T;candidates=[]
        specs=[('joint','joint',24,list(range(9)),None),('density','density',24,list(range(9)),'density'),('entropy','entropy',24,list(range(9)),'entropy')]
        specs += [(f'window_{r}','joint',24,[i for i,a in enumerate(ACTIONS) if a.endswith(f'r{r}')],None) for r in (4,8,16)]
        specs += [(f'iteration_{p}',f'iteration_p{p}',p,[i for i,a in enumerate(ACTIONS) if a.startswith(f'p{p}-')],None) for p in (24,32,49)]
        for name,key,p,allowed,scalar in specs:
            x=feature_matrix(cal,p)
            if scalar=='density':x=x[:,[0,1]].mean(1,keepdims=True)
            elif scalar=='entropy':x=x[:,[11]]
            start=allowed[0] if name.startswith('iteration') else 0
            yy=ce[:,start,None]-ce;yy[:,start]=0
            fit=ridge_fit(x,yy);pred=ridge_predict(fit,x);pred[:,start]=0
            pick=choose_penalties(pred,ce,costs(cal,key),L,C,T,float(ce[:,start].mean()),allowed)
            candidates.append({'name':name,'cost_key':key,'prefix':p,'scalar':scalar,'start_action':start,'ridge':fit,'selection':pick,'feasible':pick is not None})
        frozen=[v for v in candidates if v['name'] in ('joint','density','entropy')]
        for family in ('window','iteration'):
            choices=[v for v in candidates if v['name'].startswith(family+'_') and v['feasible']]
            chosen=min(choices,key=lambda v:(v['selection']['calibration_epe'],v['selection']['calibration_gpu_ms'],v['selection']['calibration_lookahead_bins'],v['prefix'],v['selection']['allowed'][0])) if choices else None
            frozen.append({'name':family,'feasible':False} if chosen is None else {**chosen,'selected_variant':chosen['name'],'name':family})
        direct=costs(cal,'direct');lp=lp_allocation(ce,direct,L,.95*C,list(range(9)),True,True)
        random=random_allocation(direct,L,.95*C)
        packet['cells'].append({'cell_id':f'L{L}-C{mult:g}','lookahead_cap':L,'gpu_cap_ms':C,'policies':frozen,'all_restricted_candidates':candidates,'fixed_mixture':lp,'random_weights':None if random is None else random[0], 'feasible_pure_actions':[ACTIONS[i] for i in range(9) if LOOK[i]<=L and direct[:,i].mean()<=.95*C]})
    return packet

def predict_action(policy,features,T):
    """Current-only CPU selector API. Feature input is the declared 14-vector."""
    if not policy['feasible']:return None
    x=np.asarray(features,dtype=np.float64).reshape(1,14)
    if policy['scalar']=='density':x=x[:,[0,1]].mean(1,keepdims=True)
    elif policy['scalar']=='entropy':x=x[:,[11]]
    pred=ridge_predict(policy['ridge'],x);pred[:,policy['start_action']]=0
    return ACTIONS[int(selected_weights(pred,policy['selection'],T).argmax(1)[0])]

def verify_calibration_environment(meta,cal,input_hashes):
    """Exact original row bytes/order and numeric runtime before packet rebuild."""
    path=Path(meta['calibration_jsonl_path'])
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest==meta['calibration_jsonl_sha256']==input_hashes[str(path)]
    saved=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    canonical=lambda v:json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)
    assert canonical(saved)==canonical(cal), 'combined table changed original calibration rows/order'
    ids=[r['sample_id'] for r in cal]
    assert hashlib.sha256(canonical(ids).encode()).hexdigest()==meta['calibration_ordered_sample_ids_sha256']
    binding=json.loads((Path(__file__).resolve().parents[1]/'configs/stage9-split-rows.json').read_text())
    assert ids==[r['sample_id'] for r in binding['rows'] if r['group']=='policy_calibration']
    thread_names=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
    threads={name:os.environ.get(name) for name in thread_names}
    assert all(v=='1' for v in threads.values())
    stream=io.StringIO()
    with contextlib.redirect_stdout(stream):np.show_config()
    runtime={'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'numpy_config_sha256':hashlib.sha256(stream.getvalue().encode()).hexdigest(),'threads':threads}
    assert runtime==meta['analysis_runtime'], 'calibration/analysis numerical environment mismatch'
    return runtime

def verify_frozen_packet(loaded,reconstructed):
    """Bind all frozen policy contents, rather than merely observed action agreement."""
    canonical=lambda v:json.dumps(v,default=serial,sort_keys=True,separators=(',',':'),allow_nan=False)
    assert canonical(loaded)==canonical(reconstructed), 'frozen policy packet differs from calibration-only reconstruction'
    return True

def actual_policy_result(dev,cell_id,policy):
    """Use executed chosen paths for deployed metrics, retaining emulation separately."""
    expected=np.asarray(policy['result']['frame_weights']).argmax(1)
    rows=[];g=np.ones((len(dev),9));w=onehot(expected)
    for i,row in enumerate(dev):
        rec=row.get('deployed_policies',{}).get(cell_id,{}).get(policy['name'])
        if rec is None:return None
        assert rec['action']==ACTIONS[expected[i]], 'executed action differs from frozen selector'
        assert rec['valid_pixels']==row['valid_pixels']
        for key in ('epe','bad3_percent','bad3_count','gpu_ms','CPU_selector_ms','wall_ms','parity_against_direct_max_abs'):
            assert np.isfinite(rec[key]) and rec[key]>=0
        assert rec['gpu_ms']>0
        direct=row['actions'][rec['action']];par=rec['parity']
        assert par['status']=='PASS' and par['both_finite'] is True
        assert par['atol_pixels']==1e-5 and par['rtol']==0
        diff=par['max_abs_difference'];assert np.isfinite(diff) and 0<=diff<=1e-5
        assert rec['parity_against_direct_max_abs']==diff
        assert par['sample_id']==row['sample_id'] and par['action']==rec['action']
        for k in ('actual_output_sha256','direct_output_sha256'):
            assert isinstance(par[k],str) and len(par[k])==64 and all(v in '0123456789abcdef' for v in par[k])
        assert par['direct_output_sha256']==direct['output_sha256']
        for k in ('differing_pixel_count','bad3_flip_count'):
            assert isinstance(par[k],int) and not isinstance(par[k],bool) and 0<=par[k]<=540*960
        assert isinstance(par['bitwise_equal'],bool)
        assert par['bitwise_equal']==(par['differing_pixel_count']==0)
        if par['bitwise_equal']:
            assert diff==0 and par['actual_output_sha256']==par['direct_output_sha256']
        else:
            assert par['differing_pixel_count']>0 and par['actual_output_sha256']!=par['direct_output_sha256']
            if diff==0:assert rec['epe']==direct['epe'] and rec['bad3_count']==direct['bad3_count'] and par['bad3_flip_count']==0
        vc=row['valid_pixels'];bc=rec['bad3_count'];dc=direct['bad3_count']
        assert isinstance(bc,int) and not isinstance(bc,bool) and 0<=bc<=vc
        assert abs(rec['bad3_percent']-100*bc/vc)<=1e-8
        assert par['bad3_flip_count']<=vc and abs(bc-dc)<=par['bad3_flip_count']
        if par['bitwise_equal']:assert bc==dc and par['bad3_flip_count']==0
        assert abs(rec['epe']-direct['epe'])<=diff
        v={k:rec[k] for k in ('epe','bad3_percent','bad3_count','valid_pixels')}
        rows.append({'source_id':row['source_id'],'valid_pixels':row['valid_pixels'],'actions':{a:v for a in ACTIONS}})
        g[i,:]=rec['gpu_ms']
    return reduction(rows,w,g)

def analyze(cal,dev,fdev,T):
    cells=[];ce,de=matrix(cal,'epe'),matrix(dev,'epe');directc,directd=costs(cal,'direct'),costs(dev,'direct')
    cf,df=feature_matrix(cal,24),feature_matrix(dev,24)
    for L,mult in itertools.product((8,16),(1.,1.5)):
        C=mult*T;pol=[]
        pol.append(fit_family(cal,dev,'joint',(cf,df),list(range(9)),0,T,L,C,'joint'))
        for name,ix in [('density',None),('entropy',11)]:
            x=(cf[:,[0,1]].mean(1,keepdims=True),df[:,[0,1]].mean(1,keepdims=True)) if ix is None else (cf[:,[ix]],df[:,[ix]])
            pol.append(fit_family(cal,dev,name,x,list(range(9)),0,T,L,C,name))
        for category in ('window','iteration'):
            candidates=[]
            for v in ((4,8,16) if category=='window' else (24,32,49)):
                allowed=([i for i,a in enumerate(ACTIONS) if a.endswith(f'r{v}')] if category=='window' else [i for i,a in enumerate(ACTIONS) if a.startswith(f'p{v}-')])
                p=24 if category=='window' else v;key='joint' if category=='window' else f'iteration_p{p}'
                out=fit_family(cal,dev,key,(feature_matrix(cal,p),feature_matrix(dev,p)),allowed,0 if category=='window' else allowed[0],T,L,C,f'{category}_{v}')
                candidates.append(out)
            feasible=[v for v in candidates if v['feasible']]
            chosen=min(feasible,key=lambda v:(v['calibration_selection']['calibration_epe'],v['calibration_selection']['calibration_gpu_ms'],v['calibration_selection']['calibration_lookahead_bins'],v['calibration_selection']['allowed'][0])) if feasible else None
            pol.append({'name':category,'feasible':chosen is not None,'candidates':candidates,**({k:v for k,v in chosen.items() if k!='name'} if chosen else {})})
        fixed=lp_allocation(ce,directc,L,.95*C,list(range(9)),True,True)
        pol.append({'name':'fixed_mixture','feasible':fixed['feasible'],'calibration_lp':fixed,**({'result':reduction(dev,fixed['weights'][:1].repeat(len(dev),axis=0),directd)} if fixed['feasible'] else {})})
        rand=random_allocation(directc,L,.95*C)
        pol.append({'name':'random','feasible':rand is not None,**({'result':reduction(dev,rand[:1].repeat(len(dev),axis=0),directd)} if rand is not None else {})})
        for i,a in enumerate(ACTIONS):
            if LOOK[i]<=L and directc[:,i].mean()<=.95*C:pol.append({'name':'fixed_'+a,'feasible':True,'result':reduction(dev,onehot(np.full(len(dev),i)),directd)})
        oracles={}
        for name,key,allowed,shared in [('joint','joint',list(range(9)),False),('fixed_mixture','direct',list(range(9)),True)]+[(f'window_{r}','joint',[i for i,a in enumerate(ACTIONS) if a.endswith(f'r{r}')],False) for r in (4,8,16)]+[(f'iteration_{p}',f'iteration_p{p}',[i for i,a in enumerate(ACTIONS) if a.startswith(f'p{p}-')],False) for p in (24,32,49)]:
            cost=costs(dev,key);lp=lp_allocation(de,cost,L,C,allowed,shared,tie=(name=='fixed_mixture'))
            oracles[name]={'lp':lp,**({'result':reduction(dev,lp['weights'],cost)} if lp['feasible'] else {})}
        execution_complete=True
        for pp in pol:
            if pp['name'] in ('joint','density','entropy','window','iteration') and pp['feasible']:
                actual=actual_policy_result(dev,f'L{L}-C{mult:g}',pp)
                pp['counterfactual_emulation_result']=pp['result']
                pp['actual_chosen_path_verified']=actual is not None
                if actual is None:execution_complete=False
                else:pp['result']=actual
        joint=pol[0];eligible=joint['feasible'] and joint['result']['gpu_ms']<=C and joint['result']['lookahead_bins']<=L
        comparisons={}
        if joint['feasible']:
            for other in pol[1:]:
                if other['feasible'] and other['result']['gpu_ms']<=C and other['result']['lookahead_bins']<=L:comparisons[other['name']]=compare(joint['result'],other['result'])
            if oracles['fixed_mixture']['lp']['feasible']:comparisons['fixed_mixture_oracle']=compare(joint['result'],oracles['fixed_mixture']['result'])
        headroom={}
        if oracles['joint']['lp']['feasible']:
            j=oracles['joint']['result']
            for name,other in oracles.items():
                if name!='joint' and other['lp']['feasible']:headroom[name]=bool(other['result']['epe']-j['epe']>=max(.1,.05*other['result']['epe']))
        family={}
        if joint['feasible']:
            w=np.array(joint['result']['frame_weights']);ix=w.argmax(1)
            for name,aa in [('refine',{1,2}),('observe',set(range(3,9)))]:
                ids=[i for i,v in enumerate(ix) if v in aa];family[name]={'frames':len(ids),'scenes':len({dev[i]['source_id'] for i in ids})}
        passed=bool(execution_complete and eligible and comparisons and all(v['pass'] for v in comparisons.values()) and headroom and all(headroom.values()) and family and all(v['frames']>=6 and v['scenes']>=3 for v in family.values()))
        cells.append({'lookahead_cap':L,'gpu_cap_ms':C,'policies':pol,'oracles':oracles,'comparisons':comparisons,'oracle_headroom':headroom,'action_families':family,'actual_chosen_paths_complete':execution_complete,'qualifies':passed})
    # F only has full-support actions: compare full16 directly without fabricating absent actions.
    def full_metrics(rows):
        scenes={}
        for s in sorted({r['source_id'] for r in rows}):
            a=[r['actions']['p49-r16'] for r in rows if r['source_id']==s];scenes[s]={'epe':np.mean([v['epe'] for v in a]),'bad3_percent':np.mean([v['bad3_percent'] for v in a])}
        return {'epe':float(np.mean([v['epe'] for v in scenes.values()])),'bad3_percent':float(np.mean([v['bad3_percent'] for v in scenes.values()])),'worst5_epe':float(np.mean(sorted(v['epe'] for v in scenes.values())[-5:]))}
    v,f=full_metrics(dev),full_metrics(fdev);price=v['epe']<=1.10*f['epe'] and v['bad3_percent']<=f['bad3_percent']+1 and v['worst5_epe']<=1.10*f['worst5_epe']
    good=[c for c in cells if c['qualifies']]
    return {'availability':{'V':v,'F':f,'pass':bool(price)},'cells':cells,'scientific_go_metrics':bool(price and len(good)>=2 and {c['lookahead_cap'] for c in good}=={8,16}),'GO':'PENDING_INDEPENDENT_NATIVE_AND_ARTIFACT_AUDITS','scope':'Saved-table analysis only. Metrics gate cannot establish prerequisite audit/integration PASS.'}

def serial(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    raise TypeError(type(x).__name__)

def validate_rows(cal,dev,fdev):
    bound_cohort(cal,'policy_calibration');bound_cohort(dev,'dev');bound_cohort(fdev,'dev')
    for rr,ns,nf in ((cal,20,160),(dev,30,60),(fdev,30,60)):
        assert len(rr)==nf and len({r['sample_id'] for r in rr})==nf
        assert len({r['source_id'] for r in rr})==ns
        assert all(sum(v['source_id']==sid for v in rr)==(8 if ns==20 else 2) for sid in {r['source_id'] for r in rr})
        for r in rr:
            assert r['valid_pixels']>0
            for a,v in r['actions'].items():
                assert a in ACTIONS and v['valid_pixels']==r['valid_pixels']
                assert all(np.isfinite(v[k]) for k in ('epe','bad3_percent','bad3_count','gpu_ms_direct'))
                assert v['epe']>=0 and 0<=v['bad3_count']<=r['valid_pixels'] and v['gpu_ms_direct']>0
                assert abs(v['bad3_percent']-100*v['bad3_count']/r['valid_pixels'])<=1e-8
            if r['arm']=='V':
                assert set(r['actions'])==set(ACTIONS)
                for p in (24,32,49):assert np.asarray(r['features'][f'p{p}']).shape==(14,) and np.isfinite(r['features'][f'p{p}']).all()
                for key in ('joint','density','entropy','iteration_p24','iteration_p32','iteration_p49'):
                    aa=ACTIONS if not key.startswith('iteration_') else [a for a in ACTIONS if a.startswith(key.split('_')[1]+'-')]
                    assert all(np.isfinite(r['policy_gpu_ms'][key][a]) and r['policy_gpu_ms'][key][a]>0 for a in aa)
            else:assert set(r['actions'])=={'p49-r4','p49-r8','p49-r16'}
    assert {r['sample_id'] for r in dev}=={r['sample_id'] for r in fdev}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--registration',type=Path,required=True);p.add_argument('--table',type=Path,required=True);p.add_argument('--metadata',type=Path,required=True);p.add_argument('--policy-packet',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    start=time.perf_counter();registration=json.loads(args.registration.read_text())
    assert registration['stage']=='S9_SAVED_TABLE_ANALYSIS' and registration['scientific_optimizer_updates']==0
    for path,expected in registration['input_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected
    assert registration['analyzer_sha256']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    assert {str(args.table),str(args.metadata),str(args.policy_packet)} <= set(registration['input_sha256'])
    meta=json.loads(args.metadata.read_text());assert meta['feature_names']==FEATURES
    rows=[json.loads(l) for l in args.table.read_text().splitlines() if l]
    cal=[r for r in rows if r['arm']=='V' and r['split']=='calibration'];dev=[r for r in rows if r['arm']=='V' and r['split']=='dev'];fdev=[r for r in rows if r['arm']=='F' and r['split']=='dev']
    assert [len(cal),len(dev),len(fdev)]==[160,60,60]
    for rr,ns in ((cal,20),(dev,30),(fdev,30)):
        assert len({r['sample_id'] for r in rr})==len(rr) and len({r['source_id'] for r in rr})==ns
    validate_rows(cal,dev,fdev)
    verified_runtime=verify_calibration_environment(meta,cal,registration['input_sha256'])
    loaded_packet=json.loads(args.policy_packet.read_text())
    verify_frozen_packet(loaded_packet,fit_calibration_policies(cal,float(meta['T_gpu_ms'])))
    packet_sha=hashlib.sha256(args.policy_packet.read_bytes()).hexdigest()
    assert all(r.get('policy_packet_sha256')==packet_sha for r in dev), 'Dev policy rows must bind frozen calibration packet'
    result=analyze(cal,dev,fdev,float(meta['T_gpu_ms']));result['verified_analysis_runtime']=verified_runtime;result['inputs_sha256']={str(q):hashlib.sha256(q.read_bytes()).hexdigest() for q in (args.table,args.metadata,args.policy_packet)};result['analysis_wall_seconds']=time.perf_counter()-start
    args.output.mkdir(parents=True,exist_ok=False);(args.output/'result.json').write_text(json.dumps(result,default=serial,allow_nan=False,indent=2)+'\n')

    with (args.output/'policies.csv').open('w',newline='') as stream:
        fields=['lookahead_cap','gpu_cap_ms','policy','feasible','actual_chosen_path_verified','epe','bad3_percent','pooled_bad3_percent','worst5_epe','gpu_ms','lookahead_bins']
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for cell in result['cells']:
            for policy in cell['policies']:
                writer.writerow({'lookahead_cap':cell['lookahead_cap'],'gpu_cap_ms':cell['gpu_cap_ms'],'policy':policy['name'],'feasible':policy['feasible'],'actual_chosen_path_verified':policy.get('actual_chosen_path_verified'),**{k:v for k,v in policy.get('result',{}).items() if k in fields}})
