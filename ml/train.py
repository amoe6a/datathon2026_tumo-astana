"""Explicit segment labels only; no guessing negative labels around a positive timestamp."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from ml.video import sample_video, features
from ml.manifest import validate_manifest

def train(manifest, output, threshold=.8, demo=False, rules=None):
    path=Path(manifest)
    if not 0 < threshold < 1:
        raise ValueError("threshold must be between zero and one")
    rules = rules or Path(__file__).resolve().parents[1]/"rules/catalog.json"
    data,_=validate_manifest(path,rules,demo=demo)
    records=data["videos"]
    x,y=[],[]
    for row in records:
        if row["split"] != "train":
            continue
        video=(path.parent/row["path"]).resolve()
        frames,duration=sample_video(video)
        f=features(frames)
        for segment in row["segments"]:
            if not 0 <= segment["start"] < segment["end"] <= duration+.1:
                raise ValueError("Invalid annotation interval")
        for idx,frame in enumerate(frames):
            labels={s["label"] for s in row["segments"] if s["start"] <= frame.time < s["end"]}
            if labels:
                x.append(f[idx]);y.append(labels)
    classes=sorted(set().union(*y)) if y else []
    if len(classes)<2 or "normal" not in classes:
        raise ValueError("Need explicit normal and event training segments")
    scaler=StandardScaler().fit(x)
    transformed=scaler.transform(x)
    # Independent binary targets preserve overlapping event annotations.
    models=[]
    for label in classes:
        target=np.asarray([int(label in labels) for labels in y])
        if len(np.unique(target)) != 2:
            raise ValueError("Each training class needs positive and negative samples")
        models.append(LogisticRegression(C=.3, class_weight="balanced", max_iter=1000,
                                        random_state=42).fit(transformed,target))
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output, mean=scaler.mean_, scale=scaler.scale_,
                        coef=np.concatenate([m.coef_ for m in models]),
                        intercept=np.concatenate([m.intercept_ for m in models]),
                        labels=np.asarray(classes), threshold=threshold,demo_only=demo,multilabel=True)
    return {"training_frames":len(y),"classes":classes,"demo_only":demo,"multilabel":True,
            "training_manifest_sha256":hashlib.sha256(path.read_bytes()).hexdigest()}

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("manifest");p.add_argument("--output",default="models/reference.npz")
    p.add_argument("--threshold",type=float,default=.8);p.add_argument("--demo",action="store_true")
    p.add_argument("--rules",default="rules/catalog.json")
    a=p.parse_args()
    if not 0<a.threshold<1:
        p.error("threshold must be between zero and one")
    result=train(a.manifest,a.output,a.threshold,a.demo,a.rules)
    Path(a.output).with_suffix(".json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({"training_frames":result["training_frames"],"class_count":len(result["classes"]),"demo_only":a.demo}))
