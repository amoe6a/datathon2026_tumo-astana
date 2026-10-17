"""Explicit segment labels only; no guessing negative labels around a positive timestamp."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from ml.video import sample_video, features

def train(manifest, output, threshold=.8, demo=False):
    path=Path(manifest)
    data=json.loads(path.read_text(encoding="utf-8"))
    records=data["videos"]
    # A recording/session group cannot straddle training and holdout sets.
    groups={}
    hashes={}
    x,y=[],[]
    for row in records:
        if row["split"] not in {"train","val","test"}:
            raise ValueError("split must be train, val or test")
        if row["group"] in groups and groups[row["group"]] != row["split"]:
            raise ValueError("Group leakage across splits")
        groups[row["group"]]=row["split"]
        video=(path.parent/row["path"]).resolve()
        digest=hashlib.sha256(video.read_bytes()).hexdigest()
        if digest in hashes and hashes[digest] != row["split"]:
            raise ValueError("Duplicate video leakage across splits")
        hashes[digest]=row["split"]
        if row["split"] != "train":
            continue
        frames,duration=sample_video(video)
        f=features(frames)
        for segment in row["segments"]:
            if not 0 <= segment["start"] < segment["end"] <= duration+.1:
                raise ValueError("Invalid annotation interval")
        for idx,frame in enumerate(frames):
            labels={s["label"] for s in row["segments"] if s["start"] <= frame.time < s["end"]}
            if len(labels)>1:
                raise ValueError("Reference baseline requires non-overlapping segment labels")
            if labels:
                x.append(f[idx]);y.append(labels.pop())
    if len(set(y))<2 or "normal" not in y:
        raise ValueError("Need explicit normal and event training segments")
    scaler=StandardScaler().fit(x)
    model=LogisticRegression(C=.3, class_weight="balanced", max_iter=1000, random_state=42).fit(scaler.transform(x),y)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output, mean=scaler.mean_, scale=scaler.scale_, coef=model.coef_, intercept=model.intercept_,
                        labels=model.classes_, threshold=threshold,demo_only=demo)
    return {"training_frames":len(y),"classes":model.classes_.tolist(),"demo_only":demo,
            "training_manifest_sha256":hashlib.sha256(path.read_bytes()).hexdigest()}

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("manifest");p.add_argument("--output",default="models/reference.npz")
    p.add_argument("--threshold",type=float,default=.8);p.add_argument("--demo",action="store_true")
    a=p.parse_args()
    if not 0<a.threshold<1:
        p.error("threshold must be between zero and one")
    result=train(a.manifest,a.output,a.threshold,a.demo)
    Path(a.output).with_suffix(".json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({"training_frames":result["training_frames"],"class_count":len(result["classes"]),"demo_only":a.demo}))
