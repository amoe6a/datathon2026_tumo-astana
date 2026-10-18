"""Local, metadata-only annotation audit; stdout contains aggregates, never row data.

Example: python -m scripts.audit_qwen_annotations --help
The output is an audit manifest, NOT a drop-in training dataset. Unknown review
states must be resolved by a human before any training/evaluation consumer uses it.
"""

import argparse
import json
import math
import os
import re
import subprocess
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

from scripts.prepare_data import parse_time
from scripts.prepare_qwen_data import digest, workbook_rows

EXPECTED_SPLITS = {"train": 26, "val": 7, "test": 7}
HEADERS = ["наименование файла", "таймстемп", "нарушение / без нарушения"]
LABELS = {"нарушение": 1, "без нарушения": 0}
BASE_FIELDS = {"path", "sha256", "split", "binary_label", "anchors", "annotations", "source_rows"}


class AuditError(ValueError):
    """Only fixed, non-private error codes may cross the CLI boundary."""


def read_baseline(path, expected_splits=None):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    videos = data.get("videos", [])
    if data.get("schema_version") != 1 or not videos:
        raise AuditError("unsupported_baseline_schema")
    grouped = {}
    for row in videos:
        if not BASE_FIELDS <= row.keys():
            raise AuditError("unsupported_baseline_fields")
        sha = row["sha256"]
        if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha):
            raise AuditError("invalid_baseline_hash")
        if type(row["binary_label"]) is not int or row["binary_label"] not in (0, 1):
            raise AuditError("invalid_baseline_label")
        if row["split"] not in EXPECTED_SPLITS:
            raise AuditError("invalid_baseline_split")
        if not isinstance(row["annotations"], list) or not all(isinstance(a, str) for a in row["annotations"]):
            raise AuditError("invalid_baseline_annotations")
        if not isinstance(row["anchors"], list) or any(
            type(t) not in (int, float) or not math.isfinite(t) or t < 0 for t in row["anchors"]
        ):
            raise AuditError("invalid_baseline_anchors")
        if sha in grouped:
            if (grouped[sha]["split"], grouped[sha]["binary_label"]) != (row["split"], row["binary_label"]):
                raise AuditError("baseline_duplicate_split_or_label_conflict")
            raise AuditError("baseline_must_be_unique_by_hash")
        grouped[sha] = row
    counts = dict(Counter(r["split"] for r in grouped.values()))
    if counts != (EXPECTED_SPLITS if expected_splits is None else expected_splits):
        raise AuditError("baseline_split_counts_mismatch")
    return data, grouped


def inspect_workbook(path, baseline_hash):
    """Return schema/counts only; never return headers or arbitrary cell values."""
    path = Path(path)
    sha = digest(path)
    rows = workbook_rows(path)
    if digest(path) != sha:
        raise AuditError("workbook_changed_during_inspection")
    header = rows[0] if rows else {}
    abc = [header.get(k, "").strip().lower() == name for k, name in zip("ABC", HEADERS)]
    d = bool(header.get("D", "").strip())
    active = [r for r in rows[1:] if any(str(v).strip() for v in r.values())]
    return {
        "path": str(path.resolve()), "sha256": sha, "bytes": path.stat().st_size,
        "mtime_ns": path.stat().st_mtime_ns, "matches_baseline_workbook": sha == baseline_hash,
        "schema": {"abc_headers_match": abc, "description_header_present": d,
                   "compatible_with_prepare_qwen_data": all(abc) and d,
                   "column_roles": {"A": "video_filename", "B": "timestamp", "C": "explicit_binary_label", "D": "manual_description"}},
        "annotation_rows": len(active),
        "description_present_rows": sum(bool(r.get("D", "").strip()) for r in active),
        "label_counts": dict(Counter({1: "positive", 0: "negative", None: "unknown"}.get(
            LABELS.get(r.get("C", "").strip().lower()), "unknown") for r in active)),
    }


def probe_duration(path):
    """Read duration only: no decoding, audio, GPU, frames or model imports."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=duration:format=duration", "-of", "json", str(path)],
        capture_output=True, text=True, timeout=15, check=False,
    )
    if result.returncode:
        raise AuditError("duration_probe_failed")
    metadata = json.loads(result.stdout)
    streams = metadata.get("streams", [])
    if not streams:
        raise AuditError("missing_video_stream")
    raw = streams[0].get("duration")
    if raw in (None, "N/A"):
        raw = metadata.get("format", {}).get("duration")
    value = float(raw)
    if not math.isfinite(value) or value <= 0:
        raise AuditError("invalid_duration")
    return value


def _stem(value):
    return Path(value.strip().replace("\\", "/")).stem.casefold()


def build_audit(baseline, workbook, video_dir, *, duration_probe=probe_duration,
                expected_splits=None):
    """Return private manifest and safe report; input files are never modified."""
    baseline_digest = digest(baseline)
    original, by_hash = read_baseline(baseline, expected_splits)
    workbook_info = inspect_workbook(workbook, original.get("workbook_sha256"))
    if not workbook_info["schema"]["compatible_with_prepare_qwen_data"]:
        raise AuditError("incompatible_workbook_schema")
    rows = workbook_rows(workbook)
    files = defaultdict(list)
    file_hashes = {}
    for path in sorted(Path(video_dir).rglob("*")):
        if path.is_file() and path.suffix.lower() in {".mov", ".mp4"}:
            files[path.stem.casefold()].append(path)
            file_hashes[path] = digest(path)
    issues = Counter()
    issues["ambiguous_video_stems"] = sum(len(paths) > 1 for paths in files.values())
    current = defaultdict(list)
    unmatched = []
    durations = {}
    for number, row in enumerate(rows[1:], start=2):
        if not any(str(v).strip() for v in row.values()):
            continue
        label = LABELS.get(row.get("C", "").strip().lower())
        description = row.get("D", "").strip()
        item = {"source_row": number, "filename_raw": row.get("A", ""),
                "label_raw": row.get("C", ""), "timestamp_raw": row.get("B", ""),
                "binary_label": label, "annotation": description, "issues": []}
        if label is None:
            item["issues"].append("unknown_binary_label")
        if not description:
            item["issues"].append("missing_description")
        try:
            timestamp = parse_time(row.get("B", ""))
            if timestamp is not None and (not math.isfinite(timestamp) or timestamp < 0):
                raise ValueError
            item["timestamp_sec"] = timestamp
        except (TypeError, ValueError, OverflowError):
            item["timestamp_sec"] = None
            item["issues"].append("invalid_timestamp")
        if label == 1 and item["timestamp_sec"] is None:
            item["issues"].append("positive_missing_timestamp")
        if label == 0 and item["timestamp_sec"] is not None:
            item["issues"].append("negative_has_timestamp")
        paths = files.get(_stem(row.get("A", "")), [])
        if not paths:
            item["issues"].append("missing_video_file")
        elif len(paths) != 1:
            item["issues"].append("ambiguous_video_mapping")
        else:
            path = paths[0]
            sha = file_hashes[path]
            item.update(path=str(path.resolve()), sha256=sha)
            if sha not in by_hash:
                item["issues"].append("video_not_in_frozen_split")
            else:
                current[sha].append(item)
                if label is not None and label != by_hash[sha]["binary_label"]:
                    item["issues"].append("baseline_label_disagreement")
            if sha not in durations:
                try:
                    duration = duration_probe(path)
                    if not math.isfinite(duration) or duration <= 0:
                        raise ValueError
                    durations[sha] = duration
                except (OSError, ValueError, TypeError, subprocess.SubprocessError):
                    durations[sha] = None
            duration = durations[sha]
            if duration is None:
                item["issues"].append("duration_unverified")
            elif item["timestamp_sec"] is not None and item["timestamp_sec"] > duration:
                item["issues"].append("timestamp_outside_video")
        if "sha256" not in item or item["sha256"] not in by_hash:
            unmatched.append(item)
        issues.update(item["issues"])
    changes = Counter()
    split_changes = defaultdict(Counter)
    records = []
    for sha, old in by_hash.items():
        sources = current[sha]
        labels = {r["binary_label"] for r in sources if r["binary_label"] is not None}
        descriptions = sorted({r["annotation"] for r in sources if r["annotation"]})
        anchors = sorted({r["timestamp_sec"] for r in sources if r["timestamp_sec"] is not None})
        record_issues = sorted({i for r in sources for i in r["issues"]})
        if not sources:
            record_issues.append("missing_annotation_coverage")
            issues["missing_annotation_coverage"] += 1
        if len(labels) > 1:
            record_issues.append("contradictory_duplicate_labels")
            issues["contradictory_duplicate_labels"] += 1
        if len(descriptions) > 1:
            issues["duplicate_description_variants"] += 1
        if len(sources) > 1:
            issues["duplicate_annotation_rows"] += len(sources) - 1
        known = bool(sources) and len(labels) == 1 and not any(r["issues"] for r in sources)
        state = "reviewed" if known else "unknown"
        row_changes = {
            "descriptions_changed": set(descriptions) != {a.strip() for a in old["annotations"] if a.strip()},
            "anchors_changed": anchors != sorted(set(old["anchors"])),
            "labels_changed": bool(labels) and labels != {old["binary_label"]},
            "unknown_supervision": state == "unknown",
            "uncovered_videos": not sources,
        }
        for key, changed in row_changes.items():
            changes[key] += int(changed)
            split_changes[old["split"]][key] += int(changed)
        records.append({"path": sources[0]["path"] if sources else old["path"], "sha256": sha,
                        "split": old["split"], "binary_label": next(iter(labels)) if len(labels) == 1 else None,
                        "baseline_binary_label": old["binary_label"], "anchors": anchors,
                        "annotations": descriptions, "source_rows": [r["source_row"] for r in sources],
                        "annotation_status": state, "issues": record_issues, "changes": row_changes,
                        "sources": sources})
    # Check original file integrity independently from workbook coverage/mapping.
    for old in by_hash.values():
        path = Path(old["path"])
        if not path.is_absolute():
            path = Path(baseline).parent / path
        if not path.is_file():
            issues["baseline_file_missing"] += 1
        elif file_hashes.get(path, None) != old["sha256"] and digest(path) != old["sha256"]:
            issues["baseline_file_hash_mismatch"] += 1
    if digest(baseline) != baseline_digest or digest(workbook) != workbook_info["sha256"]:
        raise AuditError("input_changed_during_audit")
    review_required = any(v for k, v in issues.items() if k not in {
        "duplicate_annotation_rows", "duplicate_description_variants", "ambiguous_video_stems"})
    report = {
        "audit_schema_version": 1, "baseline_schema_version": original["schema_version"],
        "baseline_sha256": baseline_digest, "baseline_workbook_sha256": original.get("workbook_sha256"),
        "selected_workbook": workbook_info, "workbook_bytes_changed": workbook_info["sha256"] != original.get("workbook_sha256"),
        "unique_videos": len(records), "splits": dict(Counter(r["split"] for r in records)),
        "splits_preserved_by_content_hash": True, "annotation_rows": workbook_info["annotation_rows"],
        "mapped_annotation_rows": sum(len(r) for r in current.values()), "unmatched_rows": len(unmatched),
        "changes": dict(changes), "changes_by_split": {k: dict(v) for k, v in split_changes.items()},
        "issues": dict(issues), "duration_verified_unique_videos": sum(v is not None for v in durations.values()),
        "status": "needs_local_review" if review_required else "audited",
        "training_ready": False,
        "compatibility": "Audit-only schema; not a drop-in replacement for Qwen training schema v1.",
        "limitations": ["No semantic interpretation of descriptions or model evaluation.",
                        "Missing descriptions/invalid labels are unknown supervision, never negative labels.",
                        "Exact content-hash splits preserved; session independence unverified.",
                        "Previously used test split is not a pristine new benchmark.",
                        "Timestamps checked for syntax and clip bounds, not semantic correctness."],
    }
    manifest = {"schema_version": 2, "kind": "qwen_annotation_version_audit", "privacy": "local only",
                "created_at": datetime.now(UTC).isoformat(),
                "baseline_sha256": baseline_digest, "workbook_sha256": workbook_info["sha256"],
                "split_freeze_sha256": original.get("split_freeze_sha256"),
                "training_ready": False, "videos": records, "unmatched_rows": unmatched, "report": report}
    return manifest, report


def _write_private(path, value):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def write_audit(output, manifest, report):
    """Require a fresh ignored runtime directory, private from its creation."""
    runtime = Path(__file__).resolve().parents[1] / "runtime"
    output = Path(output).resolve()
    if not output.is_relative_to(runtime) or output == runtime:
        raise AuditError("output_must_be_new_runtime_subdirectory")
    if not output.parent.is_dir():
        raise AuditError("output_parent_missing")
    output.mkdir(mode=0o700, exist_ok=False)
    _write_private(output / "manifest.v2.json", manifest)
    _write_private(output / "summary.json", report)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="runtime/qwen-training-20261017/dataset.json")
    parser.add_argument("--workbook", required=True, help="Explicit human-selected candidate, no automatic newest-file selection")
    parser.add_argument("--expected-workbook-sha256", required=True)
    parser.add_argument("--candidate-workbook", action="append", default=[])
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--output", required=True, help="New directory inside ignored runtime/")
    args = parser.parse_args(argv)
    try:
        if digest(args.workbook) != args.expected_workbook_sha256:
            raise AuditError("selected_workbook_hash_mismatch")
        manifest, report = build_audit(args.baseline, args.workbook, args.video_dir)
        candidates = list(dict.fromkeys([args.workbook, *args.candidate_workbook]))
        report["workbook_candidates"] = [inspect_workbook(p, report["baseline_workbook_sha256"]) for p in candidates]
        if (report["selected_workbook"]["sha256"] != args.expected_workbook_sha256
                or digest(args.workbook) != args.expected_workbook_sha256
                or digest(args.baseline) != report["baseline_sha256"]):
            raise AuditError("input_changed_during_audit")
        manifest["report"] = report
        output = write_audit(args.output, manifest, report)
        print(json.dumps({**report, "output_directory": str(output)}, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:  # noqa: BLE001
        # Exception messages/tracebacks from XML, OS and ffprobe can contain private data.
        code = str(exc) if isinstance(exc, AuditError) else "audit_failed"
        print(json.dumps({"status": "failed", "error": code}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
