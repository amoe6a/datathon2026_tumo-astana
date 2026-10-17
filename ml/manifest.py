"""Manifest checks shared by training and evaluation. Diagnostics contain no footage."""
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


class ManifestError(ValueError):
    pass


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_label_digest(rows):
    labels = [{"sha256": r["sha256"], "binary_label": r["source"]["binary_label"],
               "timestamp_sec": r["source"]["timestamp_sec"]} for r in rows]
    return hashlib.sha256(json.dumps(labels, sort_keys=True).encode()).hexdigest()


def validate_frozen_split(rows, frozen):
    splits = frozen.get('splits', [])
    if frozen.get('dataset_hashes') != [r['sha256'] for r in rows] or len(splits) != len(rows):
        raise ValueError('Frozen split/data length or hash mismatch')
    if any(s not in ('train', 'val', 'test') for s in splits):
        raise ValueError('Unknown frozen split')
    if 'source_labels_sha256' in frozen and frozen['source_labels_sha256'] != source_label_digest(rows):
        raise ValueError('Organizer labels changed after split freeze')
    seen = {}
    for row, split in zip(rows, splits):
        if row['sha256'] in seen and seen[row['sha256']] != split:
            raise ValueError('Duplicate leakage in frozen split')
        seen[row['sha256']] = split
    return splits


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_manifest(path, rule_path, *, purpose="train", demo=False, probe=True):
    """Validate every split before using any of them; drafts deliberately fail.

    Unsupported *expected* holdout events are allowed and counted as misses by
    the evaluator. Unsupported training targets cannot produce a usable model.
    Legacy synthetic manifests need no human-review flags.
    """
    from ml.engine import load_rules
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("videos")
    if not isinstance(rows, list) or not rows:
        raise ManifestError("Manifest must contain a nonempty videos list")
    rules = load_rules(rule_path, demo)
    groups, digests, paths = {}, {}, set()
    unsupported = Counter()
    for index, row in enumerate(rows):
        def fail(message):
            raise ManifestError(f"Record {index + 1}: {message}")
        if not isinstance(row, dict):
            fail("record must be an object")
        split, group = row.get("split"), row.get("group")
        if split not in {"train", "val", "test"}:
            fail("split must be train, val or test; finish human grouping first")
        if not isinstance(group, str) or not group.strip():
            fail("a reviewed recording group is required")
        if group in groups and groups[group] != split:
            fail("group leakage across splits")
        groups[group] = split
        if not demo and row.get("group_reviewed") is not True:
            fail("group_reviewed must be true after human review")
        if not isinstance(row.get("path"), str) or not row["path"]:
            fail("path must be a nonempty string")
        video = (path.parent / row["path"]).resolve()
        if not video.is_file():
            fail("video file missing")
        if video in paths:
            fail("same video path appears twice")
        paths.add(video)
        digest = sha256(video)
        if row.get("sha256") and row["sha256"] != digest:
            fail("video SHA-256 differs from inventory")
        if digest in digests and digests[digest] != split:
            fail("duplicate video leakage across splits")
        digests[digest] = split
        duration = row.get("duration_sec")
        if probe:
            # Decode all frames, without returning footage to diagnostics.
            from ml.video import sample_video
            _, duration = sample_video(video, fps=.1, width=64)
        if not finite_number(duration) or duration <= 0:
            fail("valid duration_sec or successful probing required")
        segments = row.get("segments", [])
        if not isinstance(segments, list):
            fail("segments must be a list")
        for segment in segments:
            if not isinstance(segment, dict):
                fail("segment must be an object")
            start, end, label = segment.get("start"), segment.get("end"), segment.get("label")
            if not (finite_number(start) and finite_number(end) and 0 <= start < end <= duration + .1):
                fail("invalid annotation interval")
            if not isinstance(label, str) or not label.strip():
                fail("segment label must be nonempty")
            if split == "train" and label != "normal" and label not in rules:
                fail("unsupported training class; review the rule catalog first")
        for left in segments:
            for right in segments:
                if left is right:
                    continue
                if max(left["start"], right["start"]) < min(left["end"], right["end"]):
                    if (left["label"] == "normal") != (right["label"] == "normal"):
                        fail("normal and event intervals overlap")
        if purpose == "evaluate" and split in {"val", "test"}:
            if not demo and row.get("events_reviewed") is not True:
                fail("evaluation events must be exhaustively reviewed")
            if "events" not in row:
                fail("evaluation events missing; [] means reviewed negative")
        events = row.get("events", [])
        if not isinstance(events, list):
            fail("events must be a list")
        for event in events:
            if not isinstance(event, dict):
                fail("event must be an object")
            label, timestamp = event.get("label"), event.get("time")
            if not isinstance(label, str) or not label.strip() or label == "normal":
                fail("events require a non-normal label")
            if not finite_number(timestamp) or not 0 <= timestamp <= duration:
                fail("invalid event timestamp")
            if label not in rules:
                unsupported[label] += 1
        source = row.get("source", {})
        if row.get("events_reviewed") is True and "binary_label" in source:
            if source["binary_label"] not in (0, 1) or isinstance(source["binary_label"], bool):
                fail("invalid source binary label")
            if bool(events) != bool(source["binary_label"]):
                fail("reviewed events disagree with organizer label; resolve locally")
    return data, {"videos": len(rows), "groups": len(groups),
                  "splits": dict(Counter(r["split"] for r in rows)),
                  "unsupported_expected_events": sum(unsupported.values())}
