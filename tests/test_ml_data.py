import json
from pathlib import Path

import numpy as np
import pytest

from ml.manifest import ManifestError, validate_manifest
from scripts.prepare_data import parse_time, prepare
from scripts import evaluate as evaluator


RULES = Path(__file__).resolve().parents[1] / "rules/catalog.json"


@pytest.fixture
def manifest(tmp_path):
    rows = []
    for index, split in enumerate(["train", "val", "test"]):
        video = tmp_path / f"{index}.mov"
        video.write_bytes(bytes([index]))
        rows.append({"path": video.name, "duration_sec": 10, "group": str(index),
                     "group_reviewed": True, "split": split, "events_reviewed": True,
                     "segments": [], "events": [{"label": "cashier_leaves", "time": 4}]})
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"videos": rows}))
    return path, rows


def save(path, rows):
    path.write_text(json.dumps({"videos": rows}))


@pytest.mark.parametrize("value,expected", [("12 секунда", 12), ("1минута 2 секунда", 62),
                                         ("01:02", 62), ("1:02:03", 3723), ("", None)])
def test_source_times(value, expected):
    assert parse_time(value) == expected


@pytest.mark.parametrize("value", ["01:99", "NaN", "-2", "tomorrow"])
def test_bad_source_times(value):
    with pytest.raises(ValueError):
        parse_time(value)


@pytest.mark.parametrize("mutation,reason", [
    (lambda r: r[2].update(group=r[1]["group"]), "group leakage"),
    (lambda r: r[0].update(split=None), "split must"),
    (lambda r: r[1].update(events_reviewed=False), "exhaustively reviewed"),
    (lambda r: r[0].update(group_reviewed=False), "group_reviewed"),
    (lambda r: r[1]["events"][0].update(time=float("nan")), "invalid event"),
    (lambda r: r[1]["events"][0].update(time=True), "invalid event"),
    (lambda r: r[0].update(segments=[{"start": 0, "end": 11, "label": "normal"}]), "interval"),
    (lambda r: r[0].update(segments=[{"start": 0, "end": 2, "label": "invented"}]), "unsupported training"),
    (lambda r: r[0].update(segments=[{"start": 0, "end": 2, "label": "normal"},
                                   {"start": 1, "end": 3, "label": "cashier_leaves"}]), "overlap"),
])
def test_manifest_rejects_bad_annotations(manifest, mutation, reason):
    path, rows = manifest
    mutation(rows)
    save(path, rows)
    with pytest.raises(ManifestError, match=reason):
        validate_manifest(path, RULES, purpose="evaluate", probe=False)


def test_val_test_duplicate_leakage(manifest):
    path, rows = manifest
    (path.parent / rows[2]["path"]).write_bytes((path.parent / rows[1]["path"]).read_bytes())
    with pytest.raises(ManifestError, match="duplicate video leakage"):
        validate_manifest(path, RULES, purpose="evaluate", probe=False)


def test_unsupported_holdout_event_is_retained(manifest):
    path, rows = manifest
    rows[1]["events"][0]["label"] = "unsupported_type"
    save(path, rows)
    _, report = validate_manifest(path, RULES, purpose="evaluate", probe=False)
    assert report["unsupported_expected_events"] == 1


def test_assignment_does_not_depend_on_prediction_order():
    predicted = [{"rule": {"id": "a"}, "timecode_sec": t} for t in (2, 0)]
    expected = [{"label": "a", "time": t} for t in (0, 4)]
    assert len(evaluator.match_events(predicted, expected)) == 2
    assert len(evaluator.match_events(list(reversed(predicted)), expected)) == 2
    predicted[0]["rule"]["id"] = "b"
    assert len(evaluator.match_events(predicted, expected)) == 1


@pytest.mark.parametrize("error,reason", [(MemoryError(), "oom"), (TimeoutError(), "timeout"),
    (json.JSONDecodeError("bad", "", 0), "invalid_json"), (ValueError("Video appears truncated"), "decode")])
def test_failures_count_expected_events(manifest, monkeypatch, error, reason):
    path, _ = manifest
    monkeypatch.setattr(evaluator, "validate_manifest", lambda *a, **kw: validate_manifest(*a, **{**kw, "probe": False}))
    def fail(*args):
        raise error
    monkeypatch.setattr(evaluator, "analyze", fail)
    report = evaluator.evaluate(path, "missing", "reference", RULES, "val")
    assert report["event_counts_at_internal_3s_tolerance"] == {"tp": 0, "fp": 0, "fn": 1}
    assert report["event_metrics_all"]["recall"] == 0
    assert report["video_metrics_all"]["recall"] == 0
    assert report["event_metrics_processed"]["recall"] is None
    assert report["failures_by_reason"] == {reason: 1}


def test_failed_negative_is_not_tn(manifest, monkeypatch):
    path, rows = manifest
    rows[1]["events"] = []
    save(path, rows)
    monkeypatch.setattr(evaluator, "validate_manifest", lambda *a, **kw: validate_manifest(*a, **{**kw, "probe": False}))
    def fail(*args):
        raise TimeoutError()
    monkeypatch.setattr(evaluator, "analyze", fail)
    report = evaluator.evaluate(path, "missing", "reference", RULES, "val")
    assert report["video_counts_all"]["tn"] == 0
    assert report["video_counts_all"]["failed_negative"] == 1


def test_import_refuses_overwriting_labels(tmp_path):
    with pytest.raises(FileExistsError):
        prepare(tmp_path / "missing.zip", tmp_path)


def test_multilabel_training_and_scoring(tmp_path):
    from scripts.make_demo import clip
    from ml.train import train
    from ml.engine import reference_scores, temporal_events
    from ml.video import sample_video
    video = tmp_path / "synthetic.mov"
    clip(video, [(1, 3)], duration=4)
    rules = json.loads(RULES.read_text())
    extra = dict(next(r for r in rules["rules"] if r["id"] == "demo_absent"), id="demo_extra")
    rules["rules"].append(extra)
    rule_path = tmp_path / "rules.json"
    rule_path.write_text(json.dumps(rules))
    path = tmp_path / "manifest.json"
    save(path, [{"path": video.name, "group": "synthetic", "split": "train", "segments": [
        {"start": 0, "end": 1, "label": "normal"},
        {"start": 1, "end": 3, "label": "demo_absent"},
        {"start": 2, "end": 4, "label": "demo_extra"}]}])
    target = tmp_path / "model.npz"
    report = train(path, target, demo=True, rules=rule_path)
    assert report["multilabel"] is True
    assert report["training_frames"] == 9
    with np.load(target) as model:
        artifact = {key: model[key] for key in model.files}
    # Force both event logits high to verify independent inference, without
    # claiming that tiny synthetic training data establishes real accuracy.
    artifact["coef"] = np.zeros_like(artifact["coef"])
    artifact["intercept"] = np.array([4., 4., -4.])
    np.savez(target, **artifact)
    frames, _ = sample_video(video)
    labels, scores, threshold = reference_scores(frames, target, True)
    events = temporal_events(frames, labels, scores, threshold, {"demo_absent": {}, "demo_extra": {}})
    assert {e["rule_id"] for e in events} == {"demo_absent", "demo_extra"}


def test_last_frame_and_window_boundaries(tmp_path):
    from scripts.make_demo import clip
    from ml.video import sample_video, Frame
    from ml.qwen import make_windows
    video = tmp_path / "end.mov"
    clip(video, [], duration=1)
    frames, duration = sample_video(video)
    assert frames[-1].time == pytest.approx(29 / 30, abs=.001)
    assert duration - frames[-1].time < .04
    # Odd tails in a final overlapping window must still reach the model.
    sequence = [Frame(float(t), None) for t in np.arange(0, 20, .5)]
    windows = make_windows(sequence)
    assert windows[0][0].time == 0
    assert windows[-1][-1].time == sequence[-1].time
    assert all(b[0].time <= a[-1].time for a, b in zip(windows, windows[1:]))


def test_even_small_decode_loss_is_an_error(tmp_path, monkeypatch):
    import cv2
    from scripts.make_demo import clip
    from ml.video import sample_video
    video = tmp_path / "short.mov"
    clip(video, [], duration=1)
    real_capture = cv2.VideoCapture
    class ShortCapture:
        def __init__(self, path):
            self.cap = real_capture(path)
            self.read_count = 0
        def __getattr__(self, name):
            return getattr(self.cap, name)
        def read(self):
            self.read_count += 1
            return self.cap.read() if self.read_count < 30 else (False, None)
    monkeypatch.setattr(cv2, "VideoCapture", ShortCapture)
    with pytest.raises(ValueError, match="truncated"):
        sample_video(video)


def test_sparse_windows_keep_context_and_final_frame():
    from ml.qwen import make_windows
    from ml.video import Frame
    frames = [Frame(t / 2, None) for t in range(286)]
    windows = make_windows(frames, window_sec=32, stride_sec=24, sampling_fps=.5)
    assert len(windows) == 6
    assert max(map(len, windows)) <= 18
    assert windows[-1][-1].time == 142.5
    assert all(b[0].time <= a[-1].time for a, b in zip(windows, windows[1:]))
    with pytest.raises(ValueError, match='sampling'):
        make_windows(frames, window_sec=8, stride_sec=12)


def test_video_pool_ignores_static_background():
    import numpy as np
    from scripts.video_baseline import pooled_features
    rng = np.random.default_rng(3)
    x = rng.normal(size=(12, 4608))
    changed = x.copy()
    changed[:, :2304] += rng.normal(size=2304)
    assert np.allclose(pooled_features(x), pooled_features(changed))
    assert np.isfinite(pooled_features(x)).all()
    with pytest.raises(ValueError):
        pooled_features(x[:, :200])


def test_truncated_frozen_split_cannot_hide_unscored_videos():
    from ml.manifest import validate_frozen_split
    rows = [{'sha256': 'a'}, {'sha256': 'b'}]
    with pytest.raises(ValueError, match='length'):
        validate_frozen_split(rows, {'dataset_hashes': ['a', 'b'], 'splits': ['train']})
    with pytest.raises(ValueError, match='Duplicate'):
        validate_frozen_split([{'sha256': 'a'}, {'sha256': 'a'}],
                              {'dataset_hashes': ['a', 'a'], 'splits': ['train', 'val']})


def test_machine_proposals_stay_unreviewed_and_test_withheld(tmp_path):
    import json
    from scripts.export_pseudo_labels import export
    rows = [{'sha256': h, 'events': [], 'segments': []} for h in ['a', 'b', 'c']]
    manifest = tmp_path / 'draft.json'; manifest.write_text(json.dumps({'videos': rows}))
    split = tmp_path / 'split.json'
    split.write_text(json.dumps({'dataset_hashes': ['a', 'b', 'c'], 'splits': ['train', 'val', 'test']}))
    predictions = tmp_path / 'predictions'; predictions.mkdir()
    proposal = {'rule_id': 'cashier_leaves', 'timecode_sec': 2, 'observation': 'Synthetic proposal only.'}
    (predictions / '00.json').write_text(json.dumps({'video_sha256': 'a', 'signature': 'one',
        'status': 'done', 'events': [proposal]}))
    # A test prediction must remain inaccessible to this review export.
    (predictions / '02.json').write_text('invalid and must not be read')
    output = tmp_path / 'review.json'
    counts = export(manifest, predictions, split, output)
    result = json.loads(output.read_text())['videos']
    assert counts == {'done': 1, 'proposed_events': 1, 'missing': 1, 'withheld_test': 1}
    assert result[0]['proposed_events'] == [proposal] and result[0]['events'] == []
    assert all(not r['events_reviewed'] and not r['group_reviewed'] for r in result)
    assert result[2]['pseudo_review']['status'] == 'withheld_final_test'


def test_truncated_generation_is_an_error():
    from ml.qwen import check_generation_complete
    with pytest.raises(ValueError, match="Truncated model"):
        check_generation_complete([1, 2, 3], [99], 3)
    check_generation_complete([1, 2, 99], [99], 3)
    check_generation_complete([1, 99], 99, 3)


def test_source_evaluation_does_not_invent_exhaustive_types(tmp_path):
    from ml.manifest import sha256
    from scripts.evaluate_source_labels import evaluate_source
    rows=[]
    for i,yes in enumerate([1,0,1,0]):
        video=tmp_path/f'{i}.mov';video.write_bytes(bytes([i]))
        rows.append({'path':video.name,'sha256':sha256(video),
                     'source':{'binary_label':yes,'timestamp_sec':4 if yes else None}})
    manifest=tmp_path/'manifest.json';save(manifest,rows)
    split=tmp_path/'split.json';split.write_text(json.dumps({'dataset_hashes':[r['sha256'] for r in rows],'splits':['val']*4}))
    predictions=tmp_path/'predictions';predictions.mkdir()
    for i,events in [(0,[{'rule_id':'known','timecode_sec':4},{'rule_id':'known','timecode_sec':9}]),(1,[])]:
        (predictions/f'{i:02d}.json').write_text(json.dumps({'status':'done','video_sha256':rows[i]['sha256'],
            'signature':'same','events':events}))
    report=evaluate_source(manifest,split,predictions,'val')
    assert report['video_counts']=={'tp':1,'tn':1,'fp':0,'fn':1}
    assert report['failed_negative']==1
    assert report['anchor_recall_at_internal_3s']==.5
    assert report['unmatched_predictions_not_confirmed_false_positives']==1
    assert 'event_metrics' not in report
    assert report['processing_success_rate']==.5


def test_weak_weights_cannot_serve_api_verdicts(tmp_path):
    from ml.engine import reference_scores
    target=tmp_path/'weak.npz'
    np.savez(target,weak_labels=True)
    with pytest.raises(ValueError,match='cannot serve API verdicts'):
        reference_scores([],target,False)


def test_utf8_video_paths_with_non_utf8_system_locale(tmp_path, monkeypatch):
    import cv2
    from ml.manifest import sha256
    from scripts.weak_baseline import prepare_cache
    from scripts.evaluate_source_labels import evaluate_source

    video = tmp_path / 'синтетическая_запись.mov'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), 10, (32, 24))
    assert writer.isOpened()
    try:
        for _ in range(10):
            writer.write(np.zeros((24, 32, 3), dtype=np.uint8))
    finally:
        writer.release()
    row = {'path': video.name, 'sha256': sha256(video),
           'source': {'binary_label': 0, 'timestamp_sec': None}}
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'videos': [row]}, ensure_ascii=False), encoding='utf-8')
    split = tmp_path / 'split.json'
    split.write_text(json.dumps({'dataset_hashes': [row['sha256']], 'splits': ['val']}), encoding='utf-8')
    predictions = tmp_path / 'predictions'
    predictions.mkdir()
    (predictions / '00.json').write_text(json.dumps({
        'status': 'done', 'video_sha256': row['sha256'], 'events': []}), encoding='utf-8')

    # Reproduce a Windows cp1251 default even when this test runs on Linux.
    original_read = Path.read_text
    def locale_read(path, encoding=None, errors=None):
        return original_read(path, encoding=encoding or 'cp1251', errors=errors)
    monkeypatch.setattr(Path, 'read_text', locale_read)

    rows, cache = prepare_cache(manifest, tmp_path)
    assert rows[0]['path'] == video.name
    assert (cache / '00.npz').is_file()
    report = evaluate_source(manifest, split, predictions, 'val')
    assert report['video_counts'] == {'tp': 0, 'tn': 1, 'fp': 0, 'fn': 0}
    assert report['processing_success_rate'] == 1


def test_source_primary_metrics_deduplicate_files_and_retain_anchors(tmp_path):
    from ml.manifest import sha256
    from scripts.evaluate_source_labels import evaluate_source
    rows=[]
    for i,timestamp in enumerate([4,8]):
        video=tmp_path/f'{i}.mov';video.write_bytes(b'same encoded footage')
        rows.append({'path':video.name,'sha256':sha256(video),'source':{'binary_label':1,'timestamp_sec':timestamp}})
    manifest=tmp_path/'manifest.json';save(manifest,rows)
    split=tmp_path/'split.json';split.write_text(json.dumps({'dataset_hashes':[r['sha256'] for r in rows],'splits':['val','val']}))
    predictions=tmp_path/'predictions';predictions.mkdir()
    for i in (0,1):
        (predictions/f'{i:02d}.json').write_text(json.dumps({'status':'done','video_sha256':rows[i]['sha256'],
            'signature':'same','events':[{'rule_id':'known','timecode_sec':4},{'rule_id':'known','timecode_sec':8}]}))
    report=evaluate_source(manifest,split,predictions,'val')
    assert report['videos']==1
    assert report['video_counts']['tp']==1
    assert report['file_level']['video_counts']['tp']==2
    assert report['organizer_anchors']==2
    assert report['matched_anchors_at_internal_3s']==2
    assert report['predicted_class_counts_not_class_accuracy']['known']==2
    rows[1]['source']['binary_label']=0;save(manifest,rows)
    with pytest.raises(ValueError,match='Conflicting duplicate'):
        evaluate_source(manifest,split,predictions,'val')
