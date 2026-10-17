"""Provisional binary benchmark using organizer labels, not semantic annotations.

No normative rule or API detection claim is produced. Positive windows around
organizer timestamps are noisy weak targets; other positive frames stay unlabeled.
Exact duplicates stay grouped. Session/near-duplicate review remains necessary.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from scipy.special import expit
from ml.video import sample_video, features
from ml.manifest import sha256,source_label_digest,validate_frozen_split
from scripts.evaluate import metrics


def freeze_split(rows, path):
    label_hash=source_label_digest(rows)
    if path.exists():
        frozen = json.loads(path.read_text())
        validate_frozen_split(rows, frozen)
        if frozen['dataset_hashes'] != [r['sha256'] for r in rows]:
            raise ValueError('Frozen split belongs to another dataset')
        if frozen.get('source_labels_sha256',label_hash)!=label_hash:
            raise ValueError('Organizer labels changed after split freeze')
        if 'source_labels_sha256' not in frozen:
            frozen['source_labels_sha256']=label_hash
            path.write_text(json.dumps(frozen,indent=2))
        return frozen['splits']
    labels = [r['source']['binary_label'] for r in rows]
    groups = [r['sha256'] for r in rows]
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=20261017).split(np.zeros(len(rows)), labels, groups))
    assignments = ['train'] * len(rows)
    for index in folds[0][1]: assignments[index] = 'test'
    for index in folds[1][1]: assignments[index] = 'val'
    frozen = {'seed': 20261017, 'dataset_hashes': groups, 'source_labels_sha256':label_hash,'splits': assignments,
              'grouping': 'exact_sha256_only_provisional_not_session_verified',
              'warning': 'Not an independently session-reviewed generalization estimate. Do not run final test until configuration freeze.'}
    path.write_text(json.dumps(frozen, indent=2))
    return assignments


def prepare_cache(manifest, output):
    path = Path(manifest)
    rows = json.loads(path.read_text())['videos']
    cache = output/'features';cache.mkdir(exist_ok=True)
    for index, row in enumerate(rows):
        target = cache/f'{index:02}.npz'
        if target.exists():
            with np.load(target) as saved:
                if saved['sha256'].item() != row['sha256']:
                    raise ValueError('Feature cache hash mismatch')
            continue
        frames, duration = sample_video(path.parent/row['path'], fps=2)
        np.savez_compressed(target, x=features(frames), times=np.asarray([f.time for f in frames]),
                            duration=duration, sha256=row['sha256'])
    return rows, cache


def feature_view(x, variant):
    if variant == "temporal_delta":
        return x - np.median(x, axis=0, keepdims=True)
    return x


def train_candidate(rows, splits, cache, radius, c, variant="original"):
    x, y = [], []
    for index, row in enumerate(rows):
        if splits[index] != 'train': continue
        with np.load(cache/f'{index:02}.npz') as data:
            if row['source']['binary_label'] == 0:
                mask = np.ones(len(data['times']), dtype=bool)
                target = 0
            else:
                timestamp = row['source']['timestamp_sec']
                if timestamp is None: raise ValueError('Positive organizer timestamp missing')
                mask = np.abs(data['times']-timestamp) <= radius
                target = 1
            x.extend(feature_view(data['x'],variant)[mask]);y.extend([target]*int(mask.sum()))
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(C=c, class_weight='balanced',max_iter=1000,random_state=42)
    model.fit(scaler.transform(x),y)
    return scaler, model, len(y)


def predict(scaler, model, data, threshold, variant="original"):
    scores = expit(model.decision_function(scaler.transform(feature_view(data['x'],variant))))
    active = np.flatnonzero(scores >= threshold)
    groups = np.split(active, np.where(np.diff(active)>1)[0]+1)
    return [float(data['times'][g[np.argmax(scores[g])]]) for g in groups if len(g)>=2]


def score(rows,splits,cache,scaler,model,threshold,split,variant="original"):
    video=Counter(tp=0,tn=0,fp=0,fn=0);files=Counter(tp=0,tn=0,fp=0,fn=0);seen=set();event=Counter(tp=0,fp=0,fn=0);errors=[]
    for index,row in enumerate(rows):
        if splits[index]!=split:continue
        with np.load(cache/f'{index:02}.npz') as data:
            predicted=predict(scaler,model,data,threshold,variant)
        yes=bool(row['source']['binary_label']);got=bool(predicted)
        observed={'tp':int(yes and got),'tn':int(not yes and not got),
                  'fp':int(not yes and got),'fn':int(yes and not got)}
        files.update(observed)
        if row['sha256'] not in seen:
            video.update(observed);seen.add(row['sha256'])
        distance=min((abs(t-row['source']['timestamp_sec']) for t in predicted),default=float('inf')) if yes else float('inf')
        matched=yes and distance<=3
        event['tp']+=int(matched);event['fn']+=int(yes and not matched);event['fp']+=len(predicted)-int(matched)
        if matched:errors.append(distance)
    return {'unique_videos':len(seen),'files':sum(files.values()),
            'video_counts':dict(video),'video_metrics':metrics(video),
            'video_counts_files':dict(files),'video_metrics_files':metrics(files),
            'organizer_anchor_count':event['tp']+event['fn'],
            'matched_anchors_at_internal_3s':event['tp'],
            'anchor_recall_at_internal_3s':event['tp']/(event['tp']+event['fn']) if event['tp']+event['fn'] else None,
            'unmatched_predictions_not_confirmed_false_positives':event['fp'],
            'matched_time_error_median_sec':float(np.median(errors)) if errors else None}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('manifest')
    p.add_argument('--output',default='runtime/weak-baseline');p.add_argument('--final-test',action='store_true')
    p.add_argument('--variant',choices=['original','temporal_delta'],default='original')
    a=p.parse_args();output=Path(a.output);output.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();rows,cache=prepare_cache(a.manifest,output)
    splits=freeze_split(rows,output/'split.freeze.json')
    if (output/'final-test.started').exists() and not a.final_test:
        raise ValueError('Candidate is frozen after final test; do not tune it using test results')
    if a.final_test:
        config=json.loads((output/'selected.json').read_text())
        lock=output/'final-test.started'
        with lock.open('x') as stream:stream.write(sha256(output/'selected.json'))
        with np.load(output/'selected.npz') as artifact:
            class Scaler:
                def transform(self,x):return (x-artifact['mean'])/artifact['scale']
            class Model:
                def decision_function(self,x):return x@artifact['coef'][0]+artifact['intercept'][0]
            result=score(rows,splits,cache,Scaler(),Model(),config['threshold'],'test',config.get('feature_variant','original'))
        (output/'final-test.json').write_text(json.dumps(result,indent=2));print(json.dumps(result));return
    candidates=[];best=None;best_key=None
    # Small predeclared validation grid. No test predictions or labels are used for selection.
    for radius,c in ((1.,.03),(1.,.3),(2.,.03),(2.,.3)):
        scaler,model,frames=train_candidate(rows,splits,cache,radius,c,a.variant)
        for threshold in (.6,.8,.9,.95):
            report=score(rows,splits,cache,scaler,model,threshold,'val',a.variant)
            config={'feature_variant':a.variant,'radius_sec':radius,'C':c,'threshold':threshold,'training_frames':frames}
            candidates.append({'config':config,'validation':report})
            # Typed/exhaustive events are unavailable: select on independent
            # organizer binary labels. Single-anchor recall is only a tiebreak.
            key=(report['video_metrics']['f1'] or 0,report['anchor_recall_at_internal_3s'] or 0)
            if best_key is None or key>best_key:
                best_key=key;best=(config,report,scaler,model)
    config,report,scaler,model=best
    np.savez_compressed(output/'selected.npz',mean=scaler.mean_,scale=scaler.scale_,coef=model.coef_,
                        intercept=model.intercept_,labels=np.array(['normal','organizer_event']),threshold=config['threshold'],
                        demo_only=False,feature_variant=a.variant,weak_labels=True,semantic_rules_verified=False)
    (output/'selected.json').write_text(json.dumps(config,indent=2))
    result={'status':'provisional_weak_binary_baseline','split_counts':dict(Counter(splits)),
            'positive_by_split':{s:sum(r['source']['binary_label'] for r,sp in zip(rows,splits) if sp==s) for s in ('train','val','test')},
            'selected':config,'validation':report,'candidates':candidates,'elapsed_sec':round(time.monotonic()-started,3),
            'limitations':['Exact duplicates grouped; primary video metrics count unique files. Recording sessions need independent review.',
                           'Weak targets are not verified event intervals or semantic types.',
                           'Not integrated with legal rules/API; no real generalization claim.',
                           'Final test not evaluated; 3s is an internal diagnostic tolerance.',
                           'Only organizer-anchor recall is measurable; event precision/F1 is unavailable without exhaustive labels.']}
    (output/'validation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='candidates'},indent=2))


if __name__=='__main__':main()
