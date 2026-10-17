"""Local evaluation: aggregate output only; processing failures count as misses."""
import argparse
from collections import Counter, defaultdict
import json
import time
from pathlib import Path
import numpy as np
import httpx
from scipy.optimize import linear_sum_assignment
from ml.engine import analyze, load_rules
from ml.manifest import validate_manifest


def match_events(predicted, expected, tolerance=3):
    """Maximum-cardinality, minimum-time-error, type-preserving assignment."""
    if not predicted or not expected:
        return []
    n, m = len(predicted), len(expected)
    unmatched = (n + 1) * (tolerance + 1)
    cost = np.full((n, m + n), float(unmatched))
    cost[:, :m] = unmatched * 2
    for i, event in enumerate(predicted):
        for j, truth in enumerate(expected):
            distance = abs(event["timecode_sec"] - truth["time"])
            if event["rule"]["id"] == truth["label"] and distance <= tolerance:
                cost[i, j] = distance
    rows, cols = linear_sum_assignment(cost)
    return [(int(i), int(j), float(cost[i, j])) for i, j in zip(rows, cols)
            if j < m and cost[i, j] <= tolerance]


def metrics(counts):
    tp, fp, fn = (counts[k] for k in ("tp", "fp", "fn"))
    return {"precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None}


def failure_reason(error):
    # Error messages are examined locally but never included in aggregate output.
    message = str(error).lower()
    if isinstance(error, httpx.HTTPStatusError):
        body = error.response.text.lower()
        if "out of memory" in body:
            return "oom"
        return "timeout" if error.response.status_code in (408, 504) else "model_http_error"
    if isinstance(error, MemoryError) or "out of memory" in message:
        return "oom"
    if isinstance(error, (TimeoutError, httpx.TimeoutException)):
        return "timeout"
    if isinstance(error, json.JSONDecodeError):
        return "invalid_json"
    if any(s in message for s in ("decode", "truncated", "video timing", "no decodable")):
        return "decode"
    if any(s in message for s in ("unknown rule", "event schema", "observation", "observed window", "truncated model")):
        return "model_response"
    if any(s in message for s in ("weights missing", "no reference model", "no such file")):
        return "missing_weights"
    return "model_or_processing"


def evaluate(manifest, model, engine, rules, split):
    path = Path(manifest)
    # Inventory provides duration; actual decoding belongs inside the measured
    # try/except below so a truncated clip is counted as a processing failure.
    data, validation = validate_manifest(path, rules, purpose="evaluate", demo=engine == "demo", probe=False)
    selected = [r for r in data["videos"] if r["split"] == split]
    if not selected:
        raise ValueError("No videos in the requested holdout split")
    allowed = load_rules(rules, engine == "demo")
    video = Counter(tp=0, tn=0, fp=0, fn=0)
    events = Counter(tp=0, fp=0, fn=0)
    processed_events = Counter(tp=0, fp=0, fn=0)
    classes = defaultdict(lambda: Counter(tp=0, fp=0, fn=0))
    failures = Counter()
    errors, latencies = [], []
    failed_positive = failed_negative = unsupported = 0
    for row in selected:
        expected = row["events"]
        unsupported += sum(e["label"] not in allowed for e in expected)
        start = time.monotonic()
        reason = None
        try:
            result = analyze(path.parent / row["path"], model, engine, rules)
            predicted = result["violations"]
            if any(e["rule"]["id"] not in allowed for e in predicted):
                raise ValueError("Model returned an unknown rule")
        except Exception as error:
            reason = failure_reason(error)
        latency = time.monotonic() - start
        latencies.append(latency)
        if reason is None and latency > 300:
            reason = "over_300s"
        if reason:
            failures[reason] += 1
            failed_positive += int(bool(expected))
            failed_negative += int(not expected)
            for event in expected:
                classes[event["label"]]["fn"] += 1
            events["fn"] += len(expected)
            continue
        yes, got = bool(expected), bool(predicted)
        video["tp"] += int(yes and got)
        video["tn"] += int(not yes and not got)
        video["fp"] += int(not yes and got)
        video["fn"] += int(yes and not got)
        matches = match_events(predicted, expected)
        matched_pred = {i for i, _, _ in matches}
        matched_true = {j for _, j, _ in matches}
        errors.extend(distance for _, _, distance in matches)
        counts = {"tp": len(matches), "fp": len(predicted) - len(matches), "fn": len(expected) - len(matches)}
        events.update(counts)
        processed_events.update(counts)
        for i, event in enumerate(predicted):
            classes[event["rule"]["id"]]["tp" if i in matched_pred else "fp"] += 1
        for j, event in enumerate(expected):
            if j not in matched_true:
                classes[event["label"]]["fn"] += 1
    n, failed = len(selected), sum(failures.values())
    all_video = dict(video)
    all_video["fn"] += failed_positive
    all_video["failed_negative"] = failed_negative
    return {"split": split, "videos": n, "processed_videos": n - failed,
            "failed_or_over_300s": failed, "failures_by_reason": dict(failures),
            "processed_under_300s_rate": (n - failed) / n,
            "video_counts": dict(video), "video_counts_all": all_video,
            "accuracy_all_videos": (video["tp"] + video["tn"]) / n,
            "precision_processed": metrics(video)["precision"], "recall_processed": metrics(video)["recall"],
            "video_metrics_processed": metrics(video), "video_metrics_all": metrics(all_video),
            "event_counts_at_internal_3s_tolerance": dict(events), "event_metrics_all": metrics(events),
            "event_counts_processed": dict(processed_events), "event_metrics_processed": metrics(processed_events),
            "per_class": {label: {**counts, **metrics(counts)} for label, counts in sorted(classes.items())},
            "unsupported_expected_events": unsupported,
            "expected_events_all": sum(len(r["events"]) for r in selected),
            "median_matched_time_error_sec": float(np.median(errors)) if errors else None,
            "p95_processing_sec": float(np.percentile(latencies, 95)),
            "max_processing_sec": max(latencies), "manifest_checks": validation,
            "note": "3s is an internal diagnostic tolerance, not official scoring. Failed positives and all their events count as misses; failed negatives are abstentions, never TN. In-process timing is not cold public API latency. No real-data claim from synthetic fixtures."}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("manifest")
    p.add_argument("--model", default="models/reference.npz")
    p.add_argument("--engine", choices=["reference", "qwen", "bonsai", "demo"], default="reference")
    p.add_argument("--rules", default="rules/catalog.json")
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--output", default="runtime/evaluation.json")
    a = p.parse_args()
    result = evaluate(a.manifest, a.model, a.engine, a.rules, a.split)
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
