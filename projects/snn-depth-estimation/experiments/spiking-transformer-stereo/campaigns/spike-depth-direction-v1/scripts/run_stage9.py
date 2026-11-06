"""Exactly one fresh paired S9 arm; final5000 is primary, no restore/best selection.

Scheduled evaluation and final extraction remain in the same bounded allocation.
No scientific component is modified from the accepted zero-update preflight.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import time
from stage9_controls import (canonical,numerical_environment,require,save,sha)
from stage9_training_controls import validate_training


def summarize_frames(frames,actions):
    """Equal pixels/frame, frames/scene, scenes; pooled bad3 is separate."""
    result={}
    for action in actions:
        scenes={}
        for row in frames:scenes.setdefault(row["source_id"],[]).append(row["actions"][action])
        require(len(scenes)==30 and all(len(v)==2 for v in scenes.values()),"fixed60Dev/30scenes required")
        scene={sid:{k:sum(x[k] for x in values)/len(values) for k in ("epe","bad1_percent","bad3_percent","bad5_percent")} for sid,values in sorted(scenes.items())}
        metrics={"mean_scene_"+k:sum(x[k] for x in scene.values())/len(scene) for k in ("epe","bad1_percent","bad3_percent","bad5_percent")}
        metrics.update(worst5_scene_epe=sum(sorted((x["epe"] for x in scene.values()),reverse=True)[:5])/5,
            pooled_bad3_percent=100*sum(row["actions"][action]["bad3_count"] for row in frames)/sum(row["actions"][action]["valid_pixels"] for row in frames),
            valid_pixels=sum(row["actions"][action]["valid_pixels"] for row in frames),scenes=scene)
        result[action]=metrics
    return result


def main():
    begun=time.perf_counter();parser=argparse.ArgumentParser(description=__doc__)
    for name in ("config","storage-config","source-registry","registration","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args();config,registration,source,preflight=validate_training(args)
    import numpy as np
    import torch
    from stage9_native import build_model,digest_state,PREFIXES,HORIZONS,FEATURE_NAMES
    from stage9_data import NativeData,device_pairs
    from stage9_recipe import sample_order,backward_pair,optimizer_schedule,prefix_for
    from stage9_actions import metrics
    from stage9_finalize import finalize
    require(torch.cuda.is_available() and torch.cuda.device_count()==1 and "5090" in torch.cuda.get_device_name(0),"one5090 required")
    require(os.environ.get("SLURM_MEM_PER_NODE")=="32768","32GiB required")
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    random.seed(1234);np.random.seed(1234);torch.manual_seed(1234);torch.cuda.manual_seed_all(1234)
    model=build_model()
    init={"full_state_sha256":digest_state(model),"active_state_sha256":digest_state(model,True)}
    require(all(init[k]==preflight["initialization"][k] for k in init),"fresh active/full initialization differs from preflight")
    init["postbuild_rng_sha256"]=hashlib.sha256(pickle.dumps((random.getstate(),np.random.get_state(),torch.get_rng_state().numpy().tobytes(),[x.cpu().numpy().tobytes() for x in torch.cuda.get_rng_state_all()]))).hexdigest()
    require(init["postbuild_rng_sha256"]==preflight["initialization"]["postbuild_rng_sha256"],"fresh parent/adapter RNG changed")
    order=sample_order();order_sha=hashlib.sha256(canonical(order).encode()).hexdigest()
    require(order_sha==preflight["initialization"]["fit_order_sha256"],"paired training order changed")
    init.update(order=order,fit_order_sha256=order_sha,seed=1234,arm=config["arm"],fresh=True,restore_checkpoint=None)
    save(args.output/"initialization.json",init)
    fit=NativeData(args.storage_config,"model_fit");dev=NativeData(args.storage_config,"dev")
    model=model.cuda();optimizer,scheduler=optimizer_schedule(model)
    prefixes=PREFIXES if config["arm"]=="V" else (49,)
    actions=[f"p{p}-r{r}" for p in prefixes for r in HORIZONS]
    metadata={"config":config,"source_identity_sha256":source["source_identity_sha256"],"source_registry_sha256":sha(args.source_registry),
        "registration_sha256":sha(args.registration),"scientific_contract_sha256":config["scientific_contract_sha256"],
        "parity_addendum_sha256":config["parity_addendum_sha256"],"split_sha256":config["split_sha256"],
        "preflight_source_identity_sha256":preflight["source_identity_sha256"],"preflight_receipt_sha256":sha(Path(registration["preflight_receipt"])),
        "initialization":{k:v for k,v in init.items() if k!="order"},"analysis_runtime":numerical_environment(),
        "T_gpu_ms":preflight["T_gpu_ms"],"T_frozen_before_training":True,"gpu":torch.cuda.get_device_name(0),"torch":torch.__version__,
        "cuda":torch.version.cuda,"cudnn":torch.backends.cudnn.version(),"feature_names":list(FEATURE_NAMES),
        "primary":"fixed final5000; no best checkpoint selection",
        "loss_cycle_semantics":{"fit_rows":800,"updates_per_cycle":400,"complete_cycles_at_final":12,"partial_final_cycle_updates":200},"numerical_protocol":"no extra scientific training warmup; scheduled eval one reset warmup/connected16 endpoint decode; final extraction exact preflight-tested protocol",
        "cost_interpretation":"steady-state profiled graph CUDA segment times; not one-shot online throughput; warmups count in total wall/accounting",
        "slurm":{k:v for k,v in os.environ.items() if k.startswith("SLURM_")},"command":list(__import__('sys').argv)}
    save(args.output/"metadata.json",metadata);history=[]
    @torch.no_grad()
    def evaluate(update):
        started=time.perf_counter();model.eval();before=digest_state(model);frames=[]
        for index in range(len(dev)):
            item=dev[index];pairs,_=device_pairs(item,prefixes)
            target,valid=item["target"].cuda(),item["valid"].cuda()
            row={"sample_id":item["row"]["sample_id"],"source_id":item["row"]["source_id"],"actions":{}}
            for p in prefixes:
                model.direct(*pairs[p],p,16,{4,8,16});model.reset()
                predictions,state=model.direct(*pairs[p],p,16,{4,8,16})
                for r in HORIZONS:row["actions"][f"p{p}-r{r}"]=metrics(predictions[r],target,valid)
                del predictions,state;model.reset()
            frames.append(row);del pairs,item,target,valid
        require(digest_state(model)==before,"scheduled eval changed learned/BN tensors")
        record={"update":update,"samples_seen":2*update,"actions":summarize_frames(frames,actions),"per_frame":frames,
            "elapsed_seconds":time.perf_counter()-started,"all_source_valid_gt":True,"selection":"descriptive scheduled endpoint; final5000 primary"}
        history.append(record);save(args.output/"history.json",history)
        return record
    try:
        evaluate(0)
        with (args.output/"train.jsonl").open("x") as stream:
            for u in range(5000):
                step_started=time.perf_counter();indices=[order[(2*u+m)%800] for m in (0,1)]
                items=[fit[i] for i in indices];used_lr=optimizer.param_groups[0]["lr"]
                record=backward_pair(model,items,config["arm"],u)
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                require(bool(torch.isfinite(norm)),"nonfinite clipped gradient")
                optimizer.step();scheduler.step();model.reset();torch.cuda.synchronize()
                record.update(update=u+1,samples_seen=2*(u+1),indices=indices,sample_ids=[x["row"]["sample_id"] for x in items],
                    used_lr=used_lr,lr=scheduler.get_last_lr()[0],gradient_norm=float(norm),seconds=time.perf_counter()-step_started,
                    elapsed_seconds=time.perf_counter()-begun,arm=config["arm"])
                stream.write(json.dumps(record,allow_nan=False)+"\n");stream.flush();del items
                if (u+1)%50==0:save(args.output/"progress.json",{"status":"TRAINING","update":u+1,"samples_seen":2*(u+1),"elapsed_seconds":time.perf_counter()-begun})
                if u+1 in (2500,5000):
                    evaluate(u+1)
                    checkpoint={"model":model.state_dict(),"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),
                        "updates":u+1,"samples_seen":2*(u+1),"order":order,"config":config,"source_identity_sha256":source["source_identity_sha256"],
                        "registration_sha256":sha(args.registration),"primary_endpoint":5000,"selection":"fixed final; no best selection"}
                    temporary=args.output/"last.pt.part";torch.save(checkpoint,temporary);temporary.replace(args.output/"last.pt");del checkpoint
        require(scheduler.last_epoch==5000,"scheduler/update count mismatch")
        final_sha=sha(args.output/"last.pt")
        # Free training-only optimizer storage; no fresh model/weight reload or F dependency.
        optimizer.zero_grad(set_to_none=True);del optimizer,scheduler;torch.cuda.empty_cache();model.eval()
        provenance={"updates":5000,"arm":config["arm"],"presentations":10000,"final_checkpoint_sha256":final_sha,
            "source_identity_sha256":source["source_identity_sha256"],"registration_sha256":sha(args.registration),
            "config_sha256":sha(args.config),"preflight_receipt_sha256":metadata["preflight_receipt_sha256"],
            "preflight_result_sha256":metadata["preflight_receipt_sha256"],
            "split_sha256":config["split_sha256"],"initialization_full_state_sha256":init["full_state_sha256"],
            "fit_order_sha256":order_sha,"scientific_contract_sha256":config["scientific_contract_sha256"],
            "parity_addendum_sha256":config["parity_addendum_sha256"],"final_model_state_sha256":digest_state(model)}
        extracted=finalize(model,config["arm"],args.storage_config,args.output/"final-extraction",preflight["T_gpu_ms"],provenance)
        save(args.output/"completion.json",{"status":"COMPLETE_FIXED_FINAL5000","updates":5000,"presentations":10000,
            "final_checkpoint_sha256":final_sha,"source_identity_sha256":source["source_identity_sha256"],"T_gpu_ms":preflight["T_gpu_ms"],
            "extraction":extracted,"elapsed_seconds":time.perf_counter()-begun,"no_F_wait_or_additional_neural_job":True})
    except BaseException as error:
        save(args.output/"failure.json",{"status":"FAILED_STOP_NO_RETRY","error_type":type(error).__name__,"error":str(error),"elapsed_seconds":time.perf_counter()-begun})
        raise


if __name__=="__main__":main()
