"""Video-level binary experiment. No semantic classes, timestamps or legal verdicts.

Hyperparameters are chosen by train-only out-of-fold predictions. Validation is
measured once for the selected configuration; final test is deliberately absent.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from ml.manifest import validate_frozen_split
from scripts.evaluate import metrics


def pooled_features(x):
    """Temporal changes pooled into 3x4 cells; no absolute background or duration."""
    if x.ndim != 2 or x.shape[1] != 4608 or not len(x):
        raise ValueError('Expected nonempty appearance/motion feature matrix')
    appearance = x[:, :2304].reshape(-1, 24, 32, 3)
    appearance = np.abs(appearance - np.median(appearance, axis=0))
    motion = x[:, 2304:3072].reshape(-1, 24, 32, 1)
    edges = x[:, 3072:].reshape(-1, 2, 24, 32).transpose(0, 2, 3, 1)
    edges = np.abs(edges - np.median(edges, axis=0))
    signals = np.concatenate([appearance, motion, edges], axis=-1)
    cells = signals.reshape(-1, 3, 8, 4, 8, 6).mean(axis=(2, 4))
    return np.quantile(cells, [.5, .9, .99], axis=0).ravel()


def binary_score(y, predicted):
    counts = {k: int(v) for k, v in {
        'tp': np.sum((y == 1) & predicted), 'tn': np.sum((y == 0) & ~predicted),
        'fp': np.sum((y == 0) & predicted), 'fn': np.sum((y == 1) & ~predicted)}.items()}
    return {'videos': len(y), 'video_counts': counts, 'video_metrics': metrics(counts)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest')
    p.add_argument('--candidate', default='runtime/weak-baseline-delta')
    p.add_argument('--output', default='runtime/video-baseline')
    a = p.parse_args()
    root = Path(a.candidate); out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    if (out / 'validation.json').exists():
        raise ValueError('Experiment already evaluated; do not overwrite its selection history')
    rows = json.loads(Path(a.manifest).read_text())['videos']
    frozen = json.loads((root / 'split.freeze.json').read_text())
    validate_frozen_split(rows, frozen)
    xs, ys, ids = {}, {}, {}
    for split in ('train', 'val'):
        seen = set(); xs[split] = []; ys[split] = []; ids[split] = []
        for i, (row, sp) in enumerate(zip(rows, frozen['splits'])):
            if sp != split or row['sha256'] in seen:
                continue
            seen.add(row['sha256'])
            with np.load(root / 'features' / f'{i:02d}.npz') as cache:
                if cache['sha256'].item() != row['sha256']:
                    raise ValueError('Feature cache mismatch')
                xs[split].append(pooled_features(cache['x']))
            ys[split].append(row['source']['binary_label']); ids[split].append(row['sha256'])
        xs[split] = np.asarray(xs[split]); ys[split] = np.asarray(ys[split])
    x, y = xs['train'], ys['train']
    folds = list(StratifiedGroupKFold(4, shuffle=True, random_state=20261017).split(x, y, ids['train']))
    def estimator(c):
        return make_pipeline(StandardScaler(), LogisticRegression(C=c, class_weight='balanced', max_iter=1000, random_state=42))
    candidates = []
    for c in (.01, .1, 1.):
        oof = np.zeros(len(y))
        for train, holdout in folds:
            model = estimator(c).fit(x[train], y[train])
            oof[holdout] = model.predict_proba(x[holdout])[:, 1]
        for threshold in (.35, .5, .65):
            report = binary_score(y, oof >= threshold)
            candidates.append({'C': c, 'threshold': threshold, 'train_oof': report})
    selected = max(candidates, key=lambda c: (c['train_oof']['video_metrics']['f1'] or 0,
                                             c['train_oof']['video_metrics']['precision'] or 0))
    # Persist the train-only decision before looking at validation predictions.
    (out / 'selected.json').write_text(json.dumps(selected, indent=2))
    model = estimator(selected['C']).fit(x, y)
    scaler, lr = model.steps[0][1], model.steps[1][1]
    np.savez_compressed(out / 'selected.npz', mean=scaler.mean_, scale=scaler.scale_, coef=lr.coef_,
                        intercept=lr.intercept_, threshold=selected['threshold'],
                        labels=np.array(['normal', 'organizer_event']), weak_labels=True,
                        semantic_rules_verified=False, feature_variant='video_temporal_pool_v1')
    result = {'status': 'provisional_video_binary_not_semantic', 'selection': '4-fold train-only OOF',
              'selected': selected, 'train_candidates': candidates,
              'validation': binary_score(ys['val'], model.predict_proba(xs['val'])[:, 1] >= selected['threshold']),
              'implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'limitations': ['Only 7 unique validation videos; sessions not independently reviewed.',
                             'No timestamps or semantic rule types; weights rejected by normative API.',
                             'Test was not predicted or evaluated.']}
    (out / 'validation.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
