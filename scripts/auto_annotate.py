"""Local Qwen pseudo-annotation. Private observations are saved locally, never printed.

Organizer labels never enter the prompt. Outputs are NOT human ground truth and
must not be used to evaluate the teacher's semantic accuracy.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest')
    parser.add_argument('--split-file',default='runtime/weak-baseline/split.freeze.json')
    parser.add_argument('--split',choices=['train','val','test'],default='train')
    parser.add_argument('--model',default='models/qwen3-vl-2b')
    parser.add_argument('--engine',choices=['qwen','bonsai'],default='qwen')
    parser.add_argument('--smoke-pair',action='store_true')
    parser.add_argument('--rules',default='rules/catalog.json')
    parser.add_argument('--output',default='data/real/pseudo-qwen')
    parser.add_argument('--limit',type=int)
    parser.add_argument('--crop-top',type=float,default=1.0)
    parser.add_argument('--frozen-config',help='Selected configuration with inference_signature; required for final test')
    args=parser.parse_args()
    if not 0.5<=args.crop_top<=1:parser.error('crop-top must be in [0.5,1]')
    if args.split=='test' and not args.frozen_config:
        parser.error('Final test is locked until --frozen-config is supplied')
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1')
    # Explicitly deny Python network connections during processing.
    import socket,ipaddress
    original_connect=socket.socket.connect
    def restricted_connect(sock,address):
        if args.engine=='bonsai' and isinstance(address,tuple):
            try:allowed=ipaddress.ip_address(address[0]).is_loopback
            except ValueError:allowed=False
            if allowed:return original_connect(sock,address)
        raise RuntimeError('Offline inference may only connect to the configured local model')
    socket.socket.connect=restricted_connect
    from ml.engine import load_rules
    from ml.manifest import validate_frozen_split
    from ml.qwen import QwenSession,make_windows,parse_events,qwen_prompt
    from ml.video import sample_video,Frame
    from scripts.evaluate import failure_reason
    torch=None
    if args.engine=='qwen':
        import torch
    manifest=Path(args.manifest);data=json.loads(manifest.read_text());rows=data['videos']
    frozen=json.loads(Path(args.split_file).read_text())
    validate_frozen_split(rows, frozen)
    rules=load_rules(args.rules)
    prompt=qwen_prompt(rules)
    implementation={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
        (Path('ml/qwen.py'),Path('ml/video.py'),Path('ml/bonsai.py'))}
    signature=hashlib.sha256(json.dumps({'implementation':implementation,'prompt':prompt,'crop':args.crop_top,
        'engine':args.engine,'model_config_sha256':hashlib.sha256(Path(args.model).read_bytes()).hexdigest() if args.engine=='bonsai' else '78448d793a7eb2f7a987a1da76d464384aa1becd','window_sec':8,'stride_sec':6,
        'max_pixels':196608,'max_new_tokens':260 if args.engine=='qwen' else 400},sort_keys=True).encode()).hexdigest()
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    if args.split=='test':
        frozen_config=json.loads(Path(args.frozen_config).read_text())
        if frozen_config.get('inference_signature')!=signature:
            raise ValueError('Final test configuration differs from the selected frozen candidate')
        marker=output/'test.inference.started.json'
        if marker.exists():
            if json.loads(marker.read_text()).get('signature')!=signature:
                raise ValueError('Final test has already started with another configuration')
        else:marker.write_text(json.dumps({'signature':signature,'resume_only_same_configuration':True}))
    if torch is not None:torch.cuda.reset_peak_memory_stats()
    began=time.monotonic()
    if args.engine=='qwen':session=QwenSession(args.model)
    else:
        from ml.bonsai import BonsaiSession
        session=BonsaiSession(args.model)
    print(json.dumps({'phase':'local_model_loaded','elapsed_sec':round(time.monotonic()-began,2)}),flush=True)
    counts=Counter();selected=[i for i,s in enumerate(frozen['splits']) if s==args.split]
    if args.smoke_pair:
        selected=[next(i for i in selected if rows[i]['source']['binary_label']==label) for label in (1,0)]
    if args.limit:selected=selected[:args.limit]
    for n,index in enumerate(selected):
        target=output/f'{index:02d}.json';row=rows[index]
        if target.exists():
            cached=json.loads(target.read_text())
            if cached.get('signature')!=signature:raise ValueError('Annotation configuration mismatch; use a new output folder')
            if cached.get('status') == 'done':
                counts['cached_done']+=1;counts['done']+=1
                counts['predicted_positive']+=int(bool(cached['events']))
                continue
            # Retry technical failures with the same frozen configuration;
            # a cached failure must never masquerade as a successful resume.
            counts['retried_incomplete']+=1
        start=time.monotonic()
        record={'record_index':index,'video_sha256':row['sha256'],'signature':signature,
                'annotation_kind':'local_model_pseudo','human_reviewed':False,'grouping_reviewed':False,
                'model_revision':'78448d793a7eb2f7a987a1da76d464384aa1becd' if args.engine=='qwen' else 'b072e1d3b35a0a630cece372c2127528e0994386',
                'events':[],'uncertain_windows':[],'status':'processing'}
        try:
            frames,duration=sample_video(manifest.parent/row['path'])
            if args.crop_top<1:
                frames=[Frame(f.time,f.bgr[:round(f.bgr.shape[0]*args.crop_top)]) for f in frames]
            windows=make_windows(frames, **getattr(session, 'window_settings', {}))
            for window in windows:
                raw=session.infer(window,prompt)
                found,uncertain=parse_events(raw,rules,window[0].time,window[-1].time)
                if uncertain:record['uncertain_windows'].append([window[0].time,window[-1].time])
                for event in found:
                    if not any(e['rule_id']==event['rule_id'] and abs(e['timecode_sec']-event['timecode_sec'])<=3 for e in record['events']):
                        record['events'].append(event)
            record.update(status='done',duration_sec=duration,windows=len(windows))
            counts['done']+=1;counts['predicted_positive']+=int(bool(record['events']))
        except Exception as error:
            record.update(status='failed',failure_reason=failure_reason(error))
            counts['failed']+=1
        record['elapsed_sec']=round(time.monotonic()-start,3)
        record['cuda_peak_allocated_bytes']=torch.cuda.max_memory_allocated() if torch is not None else None
        record['cuda_peak_reserved_bytes']=torch.cuda.max_memory_reserved() if torch is not None else None
        target.write_text(json.dumps(record,ensure_ascii=False,indent=2))
        print(json.dumps({'completed':n+1,'total':len(selected),'split':args.split,'counts':dict(counts),
                          'elapsed_sec':round(time.monotonic()-began,2)}),flush=True)
    summary={'split':args.split,'counts':dict(counts),'signature':signature,
             'elapsed_sec':round(time.monotonic()-began,3),'human_reviewed':False,
             'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated() if torch is not None else None,
             'cuda_peak_reserved_bytes':torch.cuda.max_memory_reserved() if torch is not None else None}
    (output/f'{args.split}.summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)
    if hasattr(session,'close'):session.close()
    if counts['failed']:raise SystemExit(1)


if __name__=='__main__':main()
