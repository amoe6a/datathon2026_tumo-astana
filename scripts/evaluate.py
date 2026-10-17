"""Local evaluation: outputs aggregates only, never video/frame descriptions."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from ml.engine import analyze

def evaluate(manifest,model,engine,rules,split):
    path=Path(manifest);rows=json.loads(path.read_text(encoding="utf-8"))["videos"]
    selected=[r for r in rows if r["split"]==split]
    if not selected:raise ValueError("No videos in the requested holdout split")
    train_groups={r["group"] for r in rows if r["split"]=="train"}
    if any(r["group"] in train_groups for r in selected):raise ValueError("Holdout group leakage")
    tp=tn=fp=fn=failed=0;errors=[];latencies=[];event_tp=event_fp=event_fn=0
    for row in selected:
        expected=[e for e in row["events"] if e["label"]!="normal"]
        start=time.monotonic()
        try:result=analyze(path.parent/row["path"],model,engine,rules)
        except Exception:failed+=1;continue
        latency=time.monotonic()-start;latencies.append(latency)
        if latency>300:failed+=1;continue
        predicted=result["violations"];yes=bool(expected);got=bool(predicted)
        tp+=int(yes and got);tn+=int(not yes and not got);fp+=int(not yes and got);fn+=int(yes and not got)
        remaining=set(range(len(expected)))
        for event in predicted:
            candidates=[(abs(event["timecode_sec"]-expected[i]["time"]),i) for i in remaining if event["rule"]["id"]==expected[i]["label"]]
            if candidates and min(candidates)[0]<=3:
                distance,index=min(candidates);remaining.remove(index);event_tp+=1;errors.append(distance)
            else:event_fp+=1
        event_fn+=len(remaining)
    n=len(selected)
    return {"split":split,"videos":n,"failed_or_over_300s":failed,"processed_under_300s_rate":(n-failed)/n,
            "video_counts":{"tp":tp,"tn":tn,"fp":fp,"fn":fn},"accuracy_all_videos":(tp+tn)/n,
            "precision_processed":tp/(tp+fp) if tp+fp else None,"recall_processed":tp/(tp+fn) if tp+fn else None,
            "event_counts_at_internal_3s_tolerance":{"tp":event_tp,"fp":event_fp,"fn":event_fn},
            "median_matched_time_error_sec":float(np.median(errors)) if errors else None,
            "p95_processing_sec":float(np.percentile(latencies,95)) if latencies else None,
            "max_processing_sec":max(latencies) if latencies else None,
            "note":"3s is our diagnostic tolerance, not the official scoring tolerance. No real-data performance can be inferred from synthetic fixtures. Measure cold public API latency with smoke_api.py."}

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("manifest");p.add_argument("--model",default="models/reference.npz")
    p.add_argument("--engine",choices=["reference","qwen","demo"],default="reference")
    p.add_argument("--rules",default="rules/catalog.json");p.add_argument("--split",choices=["val","test"],default="val")
    p.add_argument("--output",default="runtime/evaluation.json");a=p.parse_args()
    result=evaluate(a.manifest,a.model,a.engine,a.rules,a.split)
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))
