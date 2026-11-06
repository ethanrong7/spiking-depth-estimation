"""One separately registered zero-update native S9 resource/integration preflight.

Scientific V/F tensors never update. Disposable scratch optimizer steps establish
native B1x2 memory/cost only. No checkpoint, policy fitting or scientific score
is selected here. T and forecast bind the later separate training release.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import time
from stage9_controls import (CONTRACT, PARITY, canonical, numerical_environment,
                             require, save, sha, validate_execution)


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "storage-config", "source-registry", "registration", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    config, registration, source = validate_execution(args, "S9-preflight")
    require(config["optimizer_updates"] == 0 and registration["scientific_optimizer_updates"] == 0, "zero scientific updates only")
    import numpy as np
    import torch
    from stage9_native import (build_model, parent, digest_state, InputPadder,
                              PREFIXES, HORIZONS, ACTIONS, FEATURE_NAMES, visible_input)
    from stage9_actions import execute_path, equivalent_flow, timed_cuda
    from stage9_data import NativeData, device_pairs
    from stage9_recipe import sample_order, backward_pair, optimizer_schedule
    import stage9_analyze as analyzer
    require(list(FEATURE_NAMES) == list(analyzer.FEATURES), "native/analyzer feature schema mismatch")
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1 and "5090" in torch.cuda.get_device_name(0), "one5090 required")
    require(os.environ.get("SLURM_MEM_PER_NODE") == "32768", "32GiB allocation required")
    require(config["T_warmups"] == 10 and config["T_repeats"] == 20 and config["scratch_optimizer_pairs"] == {"V":3,"F":1}, "registered preflight counts changed")
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    def seed():
        random.seed(1234);np.random.seed(1234);torch.manual_seed(1234);torch.cuda.manual_seed_all(1234)
    def rng():
        return hashlib.sha256(pickle.dumps((random.getstate(),np.random.get_state(),torch.get_rng_state().numpy().tobytes(),
            [x.cpu().numpy().tobytes() for x in torch.cuda.get_rng_state_all()]))).hexdigest()
    def buffer_digest(model):
        return hashlib.sha256(b"".join(n.encode()+v.detach().cpu().numpy().tobytes() for n,v in sorted(model.named_buffers()))).hexdigest()
    receipt = {"status":"RUNNING", "scientific_optimizer_updates":0, "scientific_scheduler_updates":0,
        "scientific_contract_sha256":CONTRACT,"parity_addendum_sha256":PARITY,
        "source_identity_sha256":source["source_identity_sha256"], "registration_sha256":sha(args.registration),
        "config_sha256":sha(args.config), "storage_config_sha256":sha(args.storage_config),
        "split_sha256":config["split_sha256"], "feature_names":list(FEATURE_NAMES),
        "analysis_runtime":numerical_environment(), "gpu":torch.cuda.get_device_name(0),
        "torch":torch.__version__,"cuda":torch.version.cuda,"cudnn":torch.backends.cudnn.version(),
        "slurm_job_id":os.environ["SLURM_JOB_ID"], "checks":{}, "cost_interpretation":"steady-state profiled graph CUDA costs after reset warmups; complete extraction wall includes warmups; not one-shot online throughput",
        "T_definition":"resident masked input; direct full49/r16, adapter and terminal decode; summed device segment elapsed"}
    save(args.output/"progress.json",receipt)
    try:
        seed();native=parent.build_model("minimal_lif",1234,True);parent_rng=rng()
        seed();V=build_model();v_rng=rng()
        seed();F=build_model();f_rng=rng()
        require(parent_rng == v_rng == f_rng, "adapter/build perturbed parent RNG")
        require(all(torch.equal(x,V.native.state_dict()[n]) and torch.equal(x,F.native.state_dict()[n]) for n,x in native.state_dict().items()), "parent/native initial tensor mismatch")
        require(digest_state(V)==digest_state(F) and digest_state(V,True)==digest_state(F,True), "V/F initialization mismatch")
        before = digest_state(V);native_before=digest_state(native)
        order=sample_order();indices=order[:2]
        receipt["initialization"]={"full_state_sha256":before,"active_state_sha256":digest_state(V,True),
            "parent_full_state_sha256":native_before,"all_parent_native_tensors_exact":True,"parent_RNG_exact":True,
            "postbuild_rng_sha256":v_rng,"stored_parameters":sum(p.numel() for p in V.parameters()),
            "active_parameters":sum(p.numel() for p in V.parameters() if p.requires_grad),
            "buffer_sha256":buffer_digest(V),"fit_order_sha256":hashlib.sha256(canonical(order).encode()).hexdigest(),
            "first_fit_indices":indices}
        # Only the fixed first fitting pair and first calibration frame are read.
        fitting=NativeData(args.storage_config,"model_fit");calibration=NativeData(args.storage_config,"policy_calibration")
        items=[fitting[i] for i in indices];item=calibration[0]
        receipt["native_inputs"]={"fit_sample_ids":[x["row"]["sample_id"] for x in items],"T_sample_id":item["row"]["sample_id"],
            "input_shape":[1,50,540,960],"padded_shape":[1,50,544,960],"raw_binary":True,
            "source_validity":"finite positive actual PFM; no disparity cutoff","target_bin":24}
        for x in items+[item]:
            require(bool(((x["left"]==0)|(x["left"]==1)).all()) and bool(((x["right"]==0)|(x["right"]==1)).all()),"reader must return raw binary")
        V=V.cuda().eval();native=native.cuda().eval();F=F.cpu()
        pairs,cpu25=device_pairs(item)
        bn_before=buffer_digest(V)
        with torch.no_grad():
            # Identical shape-specific numerical warmup protocol, no learned/BN changes.
            pad=InputPadder(pairs[49][0].shape,divis_by=32);nl,nr=pad.pad(*pairs[49])
            require(torch.equal(V.adapter.encode(nl,49),2*nl-1),"full adapter identity failed")
            for _ in range(2):
                native(*[nl,nr],iters=16,test_mode=True);native.update_block.reset_state()
                V.direct(*pairs[49],49,16);V.reset()
            native_prediction=pad.unpad(native(nl,nr,iters=16,test_mode=True)[1]);native.update_block.reset_state()
            wrapped=V.direct(*pairs[49],49,16)[0][16];V.reset()
            receipt["checks"]["parent_full_support_parity"]=equivalent_flow(wrapped,native_prediction)
            require(digest_state(native)==native_before and digest_state(V)==before,"numerical warmup changed scientific tensors")
            del native_prediction,wrapped,nl,nr
            native=native.cpu();del native
            # Future input values are removed before any model/device access.
            changed=item["left"].clone();changed[25:]=1-changed[25:]
            require(torch.equal(visible_input(changed,24),visible_input(item["left"],24)),"future mask leak")
            del changed
            state=V.initialize(V.features(*pairs[24],24))
            segmented=V.advance(state,4)[4];segmented8=V.advance(state,8)[8];segmented16=V.advance(state,16)[16]
            for r,prediction in ((4,segmented),(8,segmented8),(16,segmented16)):
                direct=V.direct(*pairs[24],24,r)[0][r]
                receipt["checks"][f"continuation_r{r}"]=equivalent_flow(prediction,direct)
            del state,segmented,segmented8,segmented16,direct
            current=V.direct(*pairs[24],24,4)[0][4]
            V.direct(*pairs[49],49,16);V.reset()
            after_future=V.direct(*pairs[24],24,4)[0][4]
            receipt["checks"]["future_counterfactual_reset_isolation"]=equivalent_flow(after_future,current)
            require(buffer_digest(V)==bn_before,"BN buffer leaked across paths")
            require(all(lif.membrane is None for cell in V.native.update_block.cells for lif in (cell.q_lif,cell.k_lif,cell.v_lif,cell.output_lif)),"persistent membrane after path")
            del current,after_future
            # Parent checks can heat kernels, but T has its own exact registered warmups.
            for _ in range(10):execute_path(V,pairs,cpu25,"direct",fixed_action="p49-r16")
            t=[]
            for _ in range(20):
                flow,timing=execute_path(V,pairs,cpu25,"direct",fixed_action="p49-r16");t.append(timing["gpu_ms"])
            T=float(np.median(t));receipt["T_gpu_ms"]=T;receipt["T_repeats_gpu_ms"]=t
            del flow
            costs={};parities={};extraction_started=time.perf_counter()
            # Complete native action-path table cost, with no scientific EPE readout.
            for kind in ("direct","joint","entropy"):
                for action in ACTIONS:
                    execute_path(V,pairs,cpu25,kind,fixed_action=action)  # symmetric path warmup
                    flow,timing=execute_path(V,pairs,cpu25,kind,fixed_action=action)
                    reference=V.direct(*pairs[int(action.split('-')[0][1:])],int(action.split('-')[0][1:]),int(action.split('-')[1][1:]))[0][int(action.split('-')[1][1:])]
                    parities[kind+":"+action]=equivalent_flow(flow,reference)
                    costs[kind+":"+action]=timing
                    del flow,reference
            for p in PREFIXES:
                for r in HORIZONS:
                    action=f"p{p}-r{r}"
                    execute_path(V,pairs,cpu25,"iteration",fixed_action=action,prefix=p)
                    flow,timing=execute_path(V,pairs,cpu25,"iteration",fixed_action=action,prefix=p)
                    reference=V.direct(*pairs[p],p,r)[0][r]
                    parities[f"iteration_p{p}:"+action]=equivalent_flow(flow,reference)
                    costs[f"iteration_p{p}:"+action]=timing
                    del flow,reference
            # Same current-only selector API as final extraction, using fixed tiny
            # prospective coefficients/actions rather than fitting native outcomes.
            selector_checks={}
            for kind in ("density","entropy","joint","iteration"):
                n = 1 if kind in ("density","entropy") else 14
                beta=np.zeros((1+2*n,9),dtype=np.float64);beta[0,1]=1
                probe_policy={"feasible":True,"scalar":kind if kind in ("density","entropy") else None,
                    "ridge":{"mean":np.zeros(n),"sd":np.ones(n),"beta":beta},"start_action":0,
                    "selection":{"allowed":[0,1,2] if kind=="iteration" else list(range(9)),
                        "lambda":0.,"mu":0.,"calibration_mean_action_cost":[T]*(3 if kind=="iteration" else 9)}}
                selector=lambda features,policy=probe_policy: analyzer.predict_action(policy,features,T)
                execute_path(V,pairs,cpu25,kind,selector=selector)
                flow,actual=execute_path(V,pairs,cpu25,kind,selector=selector)
                direct=V.direct(*pairs[24],24,8)[0][8]
                selector_checks[kind]={"parity":equivalent_flow(flow,direct),"declared_current_features":actual["current_features"]}
                if kind == "density":require(all(x==0 for x in actual["current_features"][2:]),"density feature leakage")
                if kind == "entropy":require(all(x==0 for i,x in enumerate(actual["current_features"]) if i!=11),"entropy feature leakage")
                del flow,direct
            receipt["checks"]["current_only_selector_paths"]=selector_checks
            receipt["native_action_costs"]=costs;receipt["checks"]["all_counterfactual_path_parity"]=parities
            receipt["extraction_probe_wall_seconds_with_warmups_and_extra_reference_calls"]=time.perf_counter()-extraction_started
            require(buffer_digest(V)==bn_before and digest_state(V)==before,"counterfactual calls changed scientific tensors")
        receipt["scientific_tensors_unchanged"]=True
        receipt["inference_peak_memory"]={"allocated_bytes":torch.cuda.max_memory_allocated(),"reserved_bytes":torch.cuda.max_memory_reserved()}
        V=V.cpu();del pairs,cpu25;torch.cuda.empty_cache()
        # Disposable model only. Scientific V/F stay on CPU and are hash-checked afterward.
        scratch=build_model().cuda();opt,scheduler=optimizer_schedule(scratch)
        torch.cuda.reset_peak_memory_stats();pair_seconds=[];scratch_checks=[]
        for arm,u in (("V",0),("V",1),("V",2),("F",0)):
            begun=time.perf_counter();record=backward_pair(scratch,items,arm,u)
            norm=torch.nn.utils.clip_grad_norm_(scratch.parameters(),1.)
            require(bool(torch.isfinite(norm)),"nonfinite clipped gradient")
            opt.step();scheduler.step();scratch.reset();torch.cuda.synchronize()
            pair_seconds.append(time.perf_counter()-begun)
            # No loss/EPE is retained as scientific evidence from this disposable fit.
            record.pop("loss");record["resource_recipe_arm"]=arm;scratch_checks.append(record)
        receipt["scratch_resource_check"]={"disposable_optimizer_steps":4,"disposable_scheduler_steps":4,
            "scientific_optimizer_steps":0,"gradient_checks":scratch_checks,"paired_update_wall_seconds":pair_seconds,
            "peak_allocated_bytes":torch.cuda.max_memory_allocated(),"peak_reserved_bytes":torch.cuda.max_memory_reserved()}
        del scratch,opt,scheduler;torch.cuda.empty_cache()
        require(digest_state(V)==before==digest_state(F),"scratch check changed scientific model")
        # Conservative accounting includes every prescribed rollout and actual chosen paths.
        # Extra direct reference calls in measured extraction inflate rather than hide cost.
        train_per_update=max(pair_seconds)
        extraction_frame=receipt["extraction_probe_wall_seconds_with_warmups_and_extra_reference_calls"]
        direct16=max(v["wall_ms"] for k,v in costs.items() if k.startswith("direct:") and k.endswith("r16"))/1000
        max_policy=max(v["wall_ms"] for v in costs.values())/1000
        io_per_frame=max(x["I_O_decode_ms"] for x in items+[item])/1000
        # Synchronized parent/path checks also measure per-frame data movement separately.
        movement=item["host_device_transfer_ms"]/1000
        V_seconds=5000*(train_per_update+2*io_per_frame)+3*60*18*(direct16+io_per_frame+movement)+220*(extraction_frame+io_per_frame+movement)+60*20*2*max_policy+config["CPU_fit_io_reserve_seconds"]
        F_seconds=5000*(train_per_update+2*io_per_frame)+3*60*6*(direct16+io_per_frame+movement)+60*6*(direct16+io_per_frame+movement)+config["CPU_fit_io_reserve_seconds"]
        preflight_seconds=time.perf_counter()-started
        forecast={"basis":"max of four scratch paired updates (three V prefix-cycle pairs plus one full-support F pair); measured complete path table including numerical warmups and extra direct parity references; max20 deployed paths/frame plus warmups; per-presentation I/O; each scheduled horizon conservatively costs a16-step run plus warmup; explicit CPU fit/checkpoint reserve",
            "paired_update_seconds_used":train_per_update,"V_raw_seconds":V_seconds,"F_raw_seconds":F_seconds,
            "V_with_20percent_margin_seconds":1.2*V_seconds,"F_with_20percent_margin_seconds":1.2*F_seconds,
            "preflight_elapsed_seconds":preflight_seconds,"S9_worst_allocated_gpu_hours":17,
            "prior_study_gpu_seconds":config["prior_study_gpu_seconds"],
            "forecast_study_gpu_seconds_with_margin":config["prior_study_gpu_seconds"]+preflight_seconds+1.2*(V_seconds+F_seconds)}
        forecast["fits_per_job_and_global_caps"]=(1.2*V_seconds<=28800 and 1.2*F_seconds<=28800
            and preflight_seconds<=3600 and forecast["forecast_study_gpu_seconds_with_margin"]<=40*3600
            and preflight_seconds+1.2*(V_seconds+F_seconds)<=17*3600)
        receipt["forecast"]=forecast
        require(forecast["fits_per_job_and_global_caps"],"native forecast exceeds fixed budget; root review required, no fallback")
        receipt["status"]="PASS";receipt["elapsed_seconds"]=time.perf_counter()-started
        save(args.output/"result.json",receipt)
    except BaseException as error:
        receipt.update(status="FAIL",error_type=type(error).__name__,error=str(error),elapsed_seconds=time.perf_counter()-started)
        save(args.output/"result.json",receipt)
        raise


if __name__ == "__main__":main()
