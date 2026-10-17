"""Binary/anchor evaluation against organizer labels, with unique-video primary metrics.

One timestamp per source record is not exhaustive typed event ground truth.
No event precision/F1 or semantic classification accuracy is claimed here.
"""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import numpy as np
from ml.manifest import sha256,validate_frozen_split
from scripts.evaluate import metrics


def evaluate_source(manifest,split_file,predictions,split):
    path=Path(manifest);rows=json.loads(path.read_text(encoding='utf-8'))['videos']
    frozen=json.loads(Path(split_file).read_text(encoding='utf-8'))
    validate_frozen_split(rows, frozen)
    hash_splits={};labels={};signatures=set();groups=defaultdict(list);results={}
    for index,(row,sp) in enumerate(zip(rows,frozen['splits'])):
        digest=sha256(path.parent/row['path'])
        if digest!=row['sha256']:raise ValueError('Video hash mismatch')
        if digest in hash_splits and hash_splits[digest]!=sp:raise ValueError('Duplicate leakage')
        label=row['source']['binary_label']
        if digest in labels and labels[digest]!=label:raise ValueError('Conflicting duplicate organizer labels')
        hash_splits[digest]=sp;labels[digest]=label
        if sp!=split:continue
        groups[digest].append(index)
        target=Path(predictions)/f'{index:02d}.json'
        result=json.loads(target.read_text(encoding='utf-8')) if target.exists() else {'status':'failed','failure_reason':'missing_prediction'}
        if target.exists() and result.get('video_sha256')!=row['sha256']:raise ValueError('Prediction hash mismatch')
        if result.get('signature'):signatures.add(result['signature'])
        if len(signatures)>1:raise ValueError('Mixed inference configurations')
        results[index]=result

    def score(sets):
        counts=Counter(tp=0,tn=0,fp=0,fn=0);failures=Counter();types=Counter();errors=[];latencies=[]
        expected_anchors=matched_anchors=unmatched_predictions=failed_negative=0
        for indexes in sets:
            # Canonical lowest source index is fixed before examining predictions.
            index=min(indexes);row=rows[index];result=results[index]
            yes=bool(row['source']['binary_label'])
            anchors=sorted({rows[i]['source']['timestamp_sec'] for i in indexes
                            if yes and rows[i]['source']['timestamp_sec'] is not None})
            expected_anchors+=len(anchors)
            if result['status']!='done':
                failures[result.get('failure_reason','unknown')]+=1
                counts['fn']+=int(yes);failed_negative+=int(not yes);continue
            if 'elapsed_sec' in result:latencies.append(result['elapsed_sec'])
            events=result['events'];got=bool(events)
            for event in events:types[event['rule_id']]+=1
            counts.update({'tp':int(yes and got),'tn':int(not yes and not got),
                           'fp':int(not yes and got),'fn':int(yes and not got)})
            matched_predictions=set()
            for anchor in anchors:
                distances=[abs(event['timecode_sec']-anchor) for event in events]
                distance=min(distances,default=float('inf'))
                if distance<=3:
                    matched_anchors+=1;errors.append(distance);matched_predictions.add(distances.index(distance))
            unmatched_predictions+=len(events)-len(matched_predictions)
        total=len(sets)
        return {'videos':total,'video_counts':dict(counts),'video_metrics':metrics(counts),
            'accuracy_all_videos':(counts['tp']+counts['tn'])/total if total else None,
            'failures_by_reason':dict(failures),'failed_negative':failed_negative,
            'processing_success_rate':(total-sum(failures.values()))/total if total else None,
            'organizer_anchors':expected_anchors,'matched_anchors_at_internal_3s':matched_anchors,
            'anchor_recall_at_internal_3s':matched_anchors/expected_anchors if expected_anchors else None,
            'median_matched_time_error_sec':float(np.median(errors)) if errors else None,
            'unmatched_predictions_not_confirmed_false_positives':unmatched_predictions,
            'predicted_class_counts_not_class_accuracy':dict(types),
            'max_warm_analysis_sec':max(latencies) if latencies else None}
    unique=score(list(groups.values()))
    file_level=score([[index] for indexes in groups.values() for index in indexes])
    return {'split':split,**unique,'file_level':file_level,'inference_signatures':sorted(signatures),
            'limitations':['Primary counts use unique video SHA256; file-level counts are separate.',
                           'Exact duplicate grouping verified; recording-session grouping not reviewed.',
                           'Distinct source timestamps on identical copies are retained as annotation marks, not assumed distinct typed events.',
                           'No exhaustive typed event ground truth: no semantic/event F1 claim.',
                           '3 seconds is internal, not official; warm timing is not API cold latency.']}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('manifest');p.add_argument('predictions')
    p.add_argument('--split-file',default='runtime/weak-baseline/split.freeze.json')
    p.add_argument('--split',choices=['val','test'],default='val');p.add_argument('--output',default='runtime/source-evaluation.json')
    p.add_argument('--frozen-config')
    a=p.parse_args();out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    if a.split=='test':
        if not a.frozen_config:p.error('test requires --frozen-config before any test inference')
        with out.with_suffix('.started').open('x') as stream:stream.write(sha256(a.frozen_config))
    report=evaluate_source(a.manifest,a.split_file,a.predictions,a.split)
    if a.split=='test':
        selected=json.loads(Path(a.frozen_config).read_text(encoding='utf-8'))
        if report['inference_signatures']!=[selected.get('inference_signature')]:
            raise ValueError('Final test predictions do not match frozen configuration')
    out.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
