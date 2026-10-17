"""Evaluate frozen local videos without sending observations or images to a cloud service."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import time

os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1')


def summarize(records):
    counts=dict(TP=0,TN=0,FP=0,FN=0)
    for r in records:
        if r['status']!='done':continue
        counts[('T' if r['expected']==r['predicted'] else 'F')+('P' if r['predicted'] else 'N')]+=1
    tp,tn,fp,fn=(counts[k] for k in ('TP','TN','FP','FN'))
    processed=tp+tn+fp+fn
    times=sorted(r['seconds'] for r in records)
    errors=[r['nearest_anchor_error_sec'] for r in records if r.get('nearest_anchor_error_sec') is not None]
    return dict(counts,attempted=len(records),processed=processed,failed=len(records)-processed,
        precision=tp/(tp+fp) if tp+fp else 0,recall=tp/(tp+fn) if tp+fn else 0,
        f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0,
        accuracy_including_failures=(tp+tn)/len(records) if records else 0,
        max_warm_seconds=max(times,default=0),
        median_nearest_anchor_error_sec=sorted(errors)[len(errors)//2] if errors else None,
        timestamp_note='Nearest organizer anchor, only for detected positive videos; not a semantic event match.',
        actor_quality_note='Free-text human annotations supervise training; role/action accuracy has not been independently scored.')


def main(args):
    from ml.qwen_rag import RagSession,load_config
    from ml.video import sample_video
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False);os.chmod(out,0o700)
    manifest=json.loads(Path(args.manifest).read_text(encoding='utf-8'))
    rows=[r for r in manifest['videos'] if r['split']==args.split]
    if args.limit:rows=rows[:args.limit]
    config=load_config(args.config)
    source_hashes={str(p.relative_to(Path(__file__).resolve().parents[1])):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [Path(__file__).resolve(),Path(__file__).resolve().parents[1]/'ml/qwen_rag.py',
                  Path(__file__).resolve().parents[1]/'ml/scene.py',Path(__file__).resolve().parents[1]/'ml/rag.py']}
    if args.adapter:config['adapter']=str(Path(args.adapter).resolve())
    started=time.monotonic();session=RagSession(config);load_seconds=time.monotonic()-started
    records=[]
    def timeout(*_):raise TimeoutError('Evaluation clip exceeded 270 seconds')
    signal.signal(signal.SIGALRM,timeout)
    for i,row in enumerate(rows):
        before=time.monotonic();record={'sha256':row['sha256'],'expected':int(row['binary_label'])}
        try:
            signal.alarm(270)
            frames,duration=sample_video(row['path'],fps=2)
            events,warnings=session.analyze(frames)
            record.update(status='done',predicted=int(bool(events)),duration_sec=round(duration,3),
                event_count=len(events),rule_ids=sorted({e['rule_id'] for e in events}),warnings=len(warnings))
            if events and row['binary_label'] and row['anchors']:
                record['nearest_anchor_error_sec']=round(min(abs(e['timecode_sec']-a) for e in events for a in row['anchors']),3)
            private={'sha256':row['sha256'],'events':events,'warnings':warnings,'observations':session.last_observations}
            target=out/f'private-{i+1:02d}.json';target.write_text(json.dumps(private,ensure_ascii=False,indent=2),encoding='utf-8');os.chmod(target,0o600)
        except Exception as exc:
            record.update(status='failed',error_type=type(exc).__name__)
            # This message is diagnostic only and must not include generated descriptions.
            if isinstance(exc,(ValueError,TimeoutError)):record['error']=str(exc)[:180]
        finally:signal.alarm(0)
        record['generation_diagnostics']=session.diagnostics
        record['seconds']=round(time.monotonic()-before,3);records.append(record)
        report={'split':args.split,'config':config,'model_load_seconds':round(load_seconds,3),
            'peak_cuda_allocated_bytes':session.torch.cuda.max_memory_allocated(),
            'manifest_sha256':hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
            'source_sha256':source_hashes,
            'summary':summarize(records),'records':records,'complete':len(records)==len(rows)}
        temp=out/'report.tmp';temp.write_text(json.dumps(report,indent=2));temp.replace(out/'report.json')
        print(json.dumps({'completed':len(records),'total':len(rows),'summary':report['summary']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--config',required=True)
    p.add_argument('--adapter');p.add_argument('--split',choices=['train','val','test'],required=True)
    p.add_argument('--output',required=True);p.add_argument('--limit',type=int,default=0)
    main(p.parse_args())
