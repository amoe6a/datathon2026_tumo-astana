"""Diagnostic binary Qwen classifier. Not a complete event/evidence detector."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import time

os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1')


def main(args):
    from ml.qwen_rag import RagSession
    from ml.scene import TRIAGE_PROMPT,select_frames,readable
    from ml.video import sample_video
    from scripts.evaluate_qwen_rag import summarize
    root=Path(__file__).resolve().parents[1]
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    config={'base_model':str(root/'models/qwen3-vl-2b'),'adapter':args.adapter,
            'rag_folder':str(root/'rules/rag'),'quantized':False,'image_side':320,'max_pixels':131072}
    source_hashes={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in
                   ['scripts/evaluate_qwen_triage.py','ml/scene.py','ml/qwen_rag.py']}
    before=time.monotonic();session=RagSession(config);load_time=time.monotonic()-before
    rows=[r for r in json.loads(Path(args.manifest).read_text())['videos'] if r['split']=='val']
    records=[]
    def timeout(*_):raise TimeoutError('Triage evaluation exceeded budget')
    signal.signal(signal.SIGALRM,timeout)
    for row in rows:
        record={'expected':int(row['binary_label']),'sha256':row['sha256']};before=time.monotonic()
        try:
            signal.alarm(120)
            frames,_=sample_video(row['path'],fps=1);selected=select_frames(frames,16)
            if not readable(selected):raise ValueError('Unreadable triage input')
            value=session.generate(selected,TRIAGE_PROMPT,8).strip().upper().rstrip('.')
            if value not in {'REVIEW','NORMAL'}:raise ValueError('Invalid triage label')
            record.update(status='done',predicted=int(value=='REVIEW'))
        except Exception as exc:record.update(status='failed',error_type=type(exc).__name__)
        finally:signal.alarm(0)
        record['seconds']=round(time.monotonic()-before,3);records.append(record)
        report={'split':'val','scope':'Binary triage only; not an event-level detector','config':config,
                'model_load_seconds':load_time,'records':records,'summary':summarize(records),
                'complete':len(records)==len(rows),'peak_cuda_allocated_bytes':session.torch.cuda.max_memory_allocated(),
                'source_sha256':source_hashes,
                'manifest_sha256':hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest()}
        (out/'report.json').write_text(json.dumps(report,indent=2))
        print(json.dumps({'completed':len(records),'summary':report['summary']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--adapter')
    p.add_argument('--output',required=True);main(p.parse_args())
