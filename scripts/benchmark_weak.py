"""Cold CPU benchmark for the provisional binary baseline, not an API/semantic score."""
import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',default='data/real/manifest.draft.json')
    parser.add_argument('--candidate',default='runtime/weak-baseline-delta')
    parser.add_argument('--output',default='runtime/weak-baseline-delta/cold-benchmark.json')
    parser.add_argument('--child-index',type=int)
    args=parser.parse_args();candidate=Path(args.candidate)
    rows=json.loads(Path(args.manifest).read_text())['videos']
    frozen=json.loads((candidate/'split.freeze.json').read_text())
    if args.child_index is not None:
        import socket
        def deny(*a,**kw):raise RuntimeError('Offline benchmark network access denied')
        socket.socket.connect=deny
        from ml.video import sample_video,features
        from scripts.weak_baseline import predict
        row=rows[args.child_index];start=time.monotonic()
        frames,duration=sample_video(Path(args.manifest).parent/row['path'])
        data={'x':features(frames),'times':np.asarray([f.time for f in frames])}
        config=json.loads((candidate/'selected.json').read_text())
        with np.load(candidate/'selected.npz') as artifact:
            class Scaler:
                def transform(self,x):return (x-artifact['mean'])/artifact['scale']
            class Model:
                def decision_function(self,x):return x@artifact['coef'][0]+artifact['intercept'][0]
            predict(Scaler(),Model(),data,config['threshold'],config.get('feature_variant','original'))
        print(json.dumps({'analysis_sec':time.monotonic()-start,
            'peak_process_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            'duration_sec':duration,'frames':len(frames)}));return
    train=[i for i,s in enumerate(frozen['splits']) if s=='train']
    index=max(train,key=lambda i:rows[i]['duration_sec']);runs=[]
    for _ in range(3):
        start=time.monotonic()
        process=subprocess.run([sys.executable,'-m','scripts.benchmark_weak','--manifest',args.manifest,
            '--candidate',args.candidate,'--child-index',str(index)],capture_output=True,text=True,timeout=240)
        if process.returncode:raise RuntimeError('Cold benchmark subprocess failed; no private stderr emitted')
        report=json.loads(process.stdout);report['cold_process_sec']=time.monotonic()-start;runs.append(report)
    report={'candidate':'provisional_weak_binary_temporal_delta','runs':runs,
            'max_cold_process_sec':max(r['cold_process_sec'] for r in runs),
            'max_peak_process_rss_bytes':max(r['peak_process_rss_bytes'] for r in runs),
            'note':'Three fresh Python processes on longest train clip. Disk cache not flushed. Python network connections forbidden. No API/semantic accuracy claim.'}
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
