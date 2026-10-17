"""Export private review proposals; never promote machine labels to reviewed truth."""
import argparse
from collections import Counter
import json
from pathlib import Path
from ml.manifest import validate_frozen_split


def export(manifest, predictions, split_file, output):
    target = Path(output)
    if target.exists():
        raise ValueError('Review draft exists; refusing to overwrite annotations')
    data = json.loads(Path(manifest).read_text()); rows = data['videos']
    if any(row.get('events_reviewed') or row.get('group_reviewed') for row in rows):
        raise ValueError('Use the unreviewed source draft; preserve existing human annotations')
    frozen = json.loads(Path(split_file).read_text()); splits = validate_frozen_split(rows, frozen)
    counts = Counter(); signatures = set()
    for index, (row, split) in enumerate(zip(rows, splits)):
        row.update(split=split, group='exact_' + row['sha256'][:16],
                   group_reviewed=False, events_reviewed=False)
        row['proposed_events'] = []
        row['pseudo_review'] = {'human_reviewed': False, 'annotation_kind': 'local_model_proposal'}
        if split == 'test':
            row['pseudo_review']['status'] = 'withheld_final_test'; counts['withheld_test'] += 1
            continue
        path = Path(predictions) / f'{index:02d}.json'
        if not path.exists():
            row['pseudo_review']['status'] = 'missing_prediction'; counts['missing'] += 1
            continue
        prediction = json.loads(path.read_text())
        if prediction['video_sha256'] != row['sha256']:
            raise ValueError('Prediction/video hash mismatch')
        signatures.add(prediction['signature'])
        if len(signatures) > 1:
            raise ValueError('Mixed inference signatures')
        row['pseudo_review'].update(status=prediction['status'], signature=prediction['signature'],
                                   uncertain_windows=prediction.get('uncertain_windows', []))
        if prediction['status'] == 'done':
            row['proposed_events'] = prediction['events']; counts['done'] += 1
            counts['proposed_events'] += len(prediction['events'])
        else:
            row['pseudo_review']['failure_reason'] = prediction.get('failure_reason', 'unknown')
            counts['failed'] += 1
    data['pseudo_annotation_warning'] = 'Machine proposals only. No exhaustive event truth, normal intervals or independent session grouping. Do not train/evaluate as reviewed labels.'
    # Keep relative video paths valid: the draft must live beside the source.
    if target.resolve().parent != Path(manifest).resolve().parent:
        raise ValueError('Review draft must be written beside the source manifest')
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    return dict(counts)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest'); p.add_argument('predictions'); p.add_argument('output')
    p.add_argument('--split-file', default='runtime/weak-baseline/split.freeze.json')
    a = p.parse_args()
    print(json.dumps(export(a.manifest, a.predictions, a.split_file, a.output)))
