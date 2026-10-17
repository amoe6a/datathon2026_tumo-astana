"""Generate synthetic geometric videos, then fit a real (but toy) local classifier."""
import json
from pathlib import Path
import cv2
import numpy as np
from ml.train import train

def clip(path, events, seed=1, duration=10):
    rng=np.random.default_rng(seed)
    w,h=480,320;fps=30
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*"mp4v"),fps,(w,h))
    if not writer.isOpened():
        raise RuntimeError("Video encoder unavailable")
    try:
        for i in range(int(duration*fps)):
            t=i/fps
            img=np.full((h,w,3),(25,32,43),dtype=np.uint8)
            cv2.rectangle(img,(55,70),(420,265),(70,79,94),-1)
            cv2.rectangle(img,(330,100),(400,180),(38,45,55),-1)
            cv2.rectangle(img,(185,135),(245,170),(90,170,110),-1)
            absent=any(s<=t<e for s,e in events)
            if not absent:
                cv2.circle(img,(140+int(5*np.sin(t*2)),200),31,(210,140,85),-1)
            noise=rng.normal(0,2,img.shape)
            img=np.clip(img.astype(float)+noise,0,255).astype(np.uint8)
            cv2.putText(img,'SYNTHETIC TEST SCENE',(18,28),cv2.FONT_HERSHEY_SIMPLEX,.65,(220,220,220),1)
            writer.write(img)
    finally:
        writer.release()

def main():
    folder=Path("data/demo");folder.mkdir(parents=True,exist_ok=True)
    videos=[]
    for i in range(6):
        events=[] if i%2==0 else [(3,6)]
        path=folder/f"train_{i}.mov";clip(path,events,i)
        segments=[{"start":0,"end":10,"label":"normal"}] if not events else [
            {"start":0,"end":3,"label":"normal"},{"start":3,"end":6,"label":"demo_absent"},{"start":6,"end":10,"label":"normal"}]
        videos.append({"path":path.name,"group":f"synthetic_{i}","split":"train","segments":segments})
    for name,events in [("normal",[]),("event",[(3,6)]),("multiple",[(2,4),(6,8)])]:
        clip(folder/f"{name}.mov",events,42)
    (folder/"manifest.json").write_text(json.dumps({"videos":videos},indent=2),encoding="utf-8")
    result=train(folder/"manifest.json","models/demo.npz",.8,True)
    print(f"Synthetic demo ready: {result['training_frames']} labeled frames. No banking accuracy claim.")

if __name__=="__main__":main()
