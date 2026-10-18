"""Synthetic-only audit fixtures. Never load event recordings in these tests."""

import json
import stat
import zipfile
from copy import deepcopy
from pathlib import Path
from xml.sax.saxutils import escape

import pytest

from scripts.audit_qwen_annotations import (
    AuditError,
    build_audit,
    inspect_workbook,
    main,
    probe_duration,
    read_baseline,
    write_audit,
)
from scripts.prepare_qwen_data import digest

HEADER = ["наименование файла", "таймстемп", "нарушение / без нарушения", "Описание"]
SPLITS = {"train": 1, "val": 1, "test": 1}


def workbook(path, rows, header=HEADER):
    xml_rows = []
    for number, values in enumerate([header, *rows], 1):
        cells = "".join(
            f'<c r="{column}{number}" t="inlineStr"><is><t>{escape(value)}</t></is></c>'
            for column, value in zip("ABCD", values) if value
        )
        xml_rows.append(f'<row r="{number}">{cells}</row>')
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("xl/worksheets/sheet1.xml",
                         '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
                         '<sheetData>' + "".join(xml_rows) + '</sheetData></worksheet>')
    return path


@pytest.fixture
def dataset(tmp_path):
    videos = tmp_path / "videos"
    videos.mkdir()
    records = []
    for index, split in enumerate(SPLITS):
        path = videos / f"synthetic-{index}.mov"
        path.write_bytes(f"synthetic bytes {index}".encode())
        records.append({"path": str(path), "sha256": digest(path), "split": split,
                        "binary_label": int(index == 0), "anchors": [2.0] if index == 0 else [],
                        "annotations": [f"SYNTHETIC PRIVATE DESCRIPTION {index}"], "source_rows": [index + 2]})
    rows = [[f"synthetic-{i}.mov", "2 сек" if i == 0 else "",
             "нарушение" if i == 0 else "без нарушения", records[i]["annotations"][0]] for i in range(3)]
    sheet = workbook(tmp_path / "annotations.xlsx", rows)
    baseline = tmp_path / "dataset.json"
    baseline.write_text(json.dumps({"schema_version": 1, "workbook_sha256": digest(sheet),
                                    "split_freeze_sha256": "f" * 64, "videos": records}), encoding="utf-8")
    return baseline, sheet, videos, rows, records


def audit(dataset, *, duration_probe=lambda path: 10.0):
    baseline, sheet, videos, _, _ = dataset
    return build_audit(baseline, sheet, videos, duration_probe=duration_probe, expected_splits=SPLITS)


def test_unchanged_schema_splits_and_original_bytes(dataset):
    baseline, _, _, _, records = dataset
    before = baseline.read_bytes()
    manifest, report = audit(dataset)
    assert baseline.read_bytes() == before
    assert report["splits"] == SPLITS
    assert report["status"] == "audited"
    assert not report["workbook_bytes_changed"]
    assert not any(report["changes"].values())
    assert {v["sha256"]: v["split"] for v in manifest["videos"]} == {r["sha256"]: r["split"] for r in records}
    assert manifest["schema_version"] == 2
    assert not manifest["training_ready"]
    assert report["selected_workbook"]["schema"]["compatible_with_prepare_qwen_data"]
    assert report["duration_verified_unique_videos"] == 3


def test_missing_description_is_unknown_not_negative(dataset):
    _, sheet, _, rows, _ = dataset
    rows[0][3] = ""
    workbook(sheet, rows)
    manifest, report = audit(dataset)
    positive = manifest["videos"][0]
    assert positive["binary_label"] == 1
    assert positive["annotation_status"] == "unknown"
    assert positive["annotations"] == []
    assert report["issues"]["missing_description"] == 1
    assert report["changes"]["unknown_supervision"] == 1
    assert report["status"] == "needs_local_review"


def test_unknown_label_does_not_become_zero(dataset):
    _, sheet, _, rows, _ = dataset
    rows[1][2] = ""
    workbook(sheet, rows)
    manifest, report = audit(dataset)
    assert manifest["videos"][1]["binary_label"] is None
    assert manifest["videos"][1]["annotation_status"] == "unknown"
    assert report["issues"]["unknown_binary_label"] == 1


def test_duplicate_content_cannot_change_split_and_labels_conflict(dataset):
    _, sheet, videos, rows, _ = dataset
    (videos / "synthetic-copy.mp4").write_bytes((videos / "synthetic-0.mov").read_bytes())
    rows.append(["synthetic-copy.mp4", "", "без нарушения", "SYNTHETIC second description"])
    workbook(sheet, rows)
    manifest, report = audit(dataset)
    positive = manifest["videos"][0]
    assert positive["split"] == "train"
    assert positive["binary_label"] is None
    assert positive["annotation_status"] == "unknown"
    assert report["issues"]["contradictory_duplicate_labels"] == 1
    assert report["issues"]["baseline_label_disagreement"] == 1
    assert report["issues"]["duplicate_annotation_rows"] == 1
    assert report["unique_videos"] == 3


def test_duplicate_anchors_retained_not_treated_as_distinct_events(dataset):
    _, sheet, videos, rows, _ = dataset
    (videos / "synthetic-copy.mov").write_bytes((videos / "synthetic-0.mov").read_bytes())
    rows.append(["synthetic-copy.mov", "3 сек", "нарушение", rows[0][3]])
    workbook(sheet, rows)
    manifest, report = audit(dataset)
    assert manifest["videos"][0]["anchors"] == [2.0, 3.0]
    assert report["changes"]["anchors_changed"] == 1
    assert report["splits"] == SPLITS


@pytest.mark.parametrize("timestamp,issue", [
    ("-1", "invalid_timestamp"), ("NaN", "invalid_timestamp"),
    ("1:99", "invalid_timestamp"), ("11 сек", "timestamp_outside_video"),
    ("", "positive_missing_timestamp"),
])
def test_timestamp_checks(dataset, timestamp, issue):
    _, sheet, _, rows, _ = dataset
    rows[0][1] = timestamp
    workbook(sheet, rows)
    manifest, report = audit(dataset)
    assert report["issues"][issue] == 1
    assert manifest["videos"][0]["annotation_status"] == "unknown"


def test_negative_timestamp_is_not_silently_accepted(dataset):
    _, sheet, _, rows, _ = dataset
    rows[1][1] = "1"
    workbook(sheet, rows)
    _, report = audit(dataset)
    assert report["issues"]["negative_has_timestamp"] == 1


def test_missing_and_unknown_files_stay_unassigned(dataset):
    _, sheet, videos, rows, _ = dataset
    (videos / "synthetic-1.mov").unlink()
    (videos / "synthetic-new.mov").write_bytes(b"unseen synthetic content")
    rows.append(["synthetic-new.mov", "", "без нарушения", "SYNTHETIC new"])
    workbook(sheet, rows)
    manifest, report = audit(dataset)
    assert len(manifest["unmatched_rows"]) == 2
    assert all("split" not in r for r in manifest["unmatched_rows"])
    assert report["issues"]["missing_video_file"] == 1
    assert report["issues"]["video_not_in_frozen_split"] == 1
    assert report["issues"]["baseline_file_missing"] == 1
    assert report["changes"]["uncovered_videos"] == 1
    assert report["splits"] == SPLITS


def test_ambiguous_stems_not_silently_selected(dataset):
    _, _, videos, _, _ = dataset
    (videos / "synthetic-0.mp4").write_bytes(b"different synthetic bytes")
    _, report = audit(dataset)
    assert report["issues"]["ambiguous_video_mapping"] == 1
    assert report["issues"]["ambiguous_video_stems"] == 1
    assert report["changes"]["uncovered_videos"] == 1


def test_modified_original_video_hash_is_detected(dataset):
    _, _, videos, _, _ = dataset
    (videos / "synthetic-1.mov").write_bytes(b"changed synthetic bytes")
    _, report = audit(dataset)
    assert report["issues"]["baseline_file_hash_mismatch"] == 1
    assert report["issues"]["video_not_in_frozen_split"] == 1


def test_description_changes_are_counted_per_split_only(dataset):
    _, sheet, _, rows, _ = dataset
    rows[2][3] = "SYNTHETIC changed description"
    workbook(sheet, rows)
    _, report = audit(dataset)
    assert report["workbook_bytes_changed"]
    assert report["changes"]["descriptions_changed"] == 1
    assert report["changes_by_split"]["test"]["descriptions_changed"] == 1
    safe = json.dumps(report)
    assert "SYNTHETIC" not in safe
    assert "synthetic-0.mov" not in safe
    assert "prediction" not in safe


def test_duration_failure_is_unknown_and_private(dataset):
    def failed(path):
        raise ValueError("SYNTHETIC PRIVATE exception content")
    manifest, report = audit(dataset, duration_probe=failed)
    assert report["issues"]["duration_unverified"] == 3
    assert report["duration_verified_unique_videos"] == 0
    assert all(r["annotation_status"] == "unknown" for r in manifest["videos"])
    assert "PRIVATE" not in json.dumps(report)


def test_schema_and_baseline_split_guards(dataset):
    baseline, sheet, _, rows, _ = dataset
    with pytest.raises(AuditError, match="baseline_split_counts_mismatch"):
        read_baseline(baseline)
    workbook(sheet, rows, header=HEADER[:3])
    assert not inspect_workbook(sheet, "f" * 64)["schema"]["compatible_with_prepare_qwen_data"]
    with pytest.raises(AuditError, match="incompatible_workbook_schema"):
        audit(dataset)


def test_baseline_duplicate_split_conflict_rejected(dataset):
    baseline, _, _, _, _ = dataset
    data = json.loads(baseline.read_text())
    duplicate = deepcopy(data["videos"][0])
    duplicate["split"] = "test"
    data["videos"].append(duplicate)
    baseline.write_text(json.dumps(data))
    with pytest.raises(AuditError, match="baseline_duplicate_split_or_label_conflict"):
        audit(dataset)


def test_private_exclusive_output_and_no_original_overwrite(dataset, tmp_path, monkeypatch):
    manifest, report = audit(dataset)
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    runtime = root / "runtime"
    runtime.mkdir()
    monkeypatch.setattr("scripts.audit_qwen_annotations.__file__", str(root / "scripts" / "audit.py"))
    output = runtime / "version-test"
    write_audit(output, manifest, report)
    assert stat.S_IMODE(output.stat().st_mode) == 0o700
    for name in ("manifest.v2.json", "summary.json"):
        assert stat.S_IMODE((output / name).stat().st_mode) == 0o600
    with pytest.raises(FileExistsError):
        write_audit(output, manifest, report)
    with pytest.raises(AuditError, match="output_must_be_new_runtime_subdirectory"):
        write_audit(tmp_path / "outside", manifest, report)
    (runtime / "escape").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(AuditError, match="output_must_be_new_runtime_subdirectory"):
        write_audit(runtime / "escape" / "version", manifest, report)


def test_cli_pin_and_exception_redaction(dataset, capsys, monkeypatch):
    baseline, sheet, videos, _, _ = dataset
    args = ["--baseline", str(baseline), "--workbook", str(sheet),
            "--video-dir", str(videos), "--output", "runtime/unused-audit",
            "--expected-workbook-sha256", "0" * 64]
    assert main(args) == 1
    assert json.loads(capsys.readouterr().out)["error"] == "selected_workbook_hash_mismatch"
    args[-1] = digest(sheet)
    def failed(*args, **kwargs):
        raise OSError("SYNTHETIC PRIVATE filename.mov and annotation")
    monkeypatch.setattr("scripts.audit_qwen_annotations.build_audit", failed)
    assert main(args) == 1
    assert json.loads(capsys.readouterr().out) == {"status": "failed", "error": "audit_failed"}


def test_workbook_changed_during_inspection_is_rejected(dataset, monkeypatch):
    from scripts import audit_qwen_annotations as module
    _, sheet, _, _, _ = dataset
    original_reader = module.workbook_rows
    def changed(path):
        rows = original_reader(path)
        with Path(path).open("ab") as stream:
            stream.write(b"synthetic concurrent change")
        return rows
    monkeypatch.setattr(module, "workbook_rows", changed)
    with pytest.raises(AuditError, match="workbook_changed_during_inspection"):
        inspect_workbook(sheet, "f" * 64)


def test_baseline_changed_during_audit_is_rejected(dataset, monkeypatch):
    from scripts import audit_qwen_annotations as module
    original_reader = module.read_baseline
    def changed(path, expected_splits):
        result = original_reader(path, expected_splits)
        with Path(path).open("a") as stream:
            stream.write(" ")
        return result
    monkeypatch.setattr(module, "read_baseline", changed)
    with pytest.raises(AuditError, match="input_changed_during_audit"):
        audit(dataset)


def test_probe_is_metadata_only_and_captures_errors(monkeypatch):
    calls = []
    def run(command, **kwargs):
        calls.append((command, kwargs))
        return type("Result", (), {"returncode": 0, "stdout": '{"streams":[{"duration":"10"}]}'} )()
    monkeypatch.setattr("scripts.audit_qwen_annotations.subprocess.run", run)
    assert probe_duration(Path("synthetic.mov")) == 10
    command, kwargs = calls[0]
    assert command[0] == "ffprobe"
    assert "-select_streams" in command and "v:0" in command
    assert "-show_frames" not in command
    assert kwargs["capture_output"] and kwargs["timeout"] == 15
