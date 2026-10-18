"""Synthetic GGUF fixtures only: no weights, footage, GPU or network access."""
import hashlib
import json
import struct

import pytest

from scripts import inspect_qwen_candidate as candidate


def u32(value):
    return struct.pack("<I", value)


def u64(value):
    return struct.pack("<Q", value)


def string(value):
    raw = value.encode("utf-8")
    return u64(len(raw)) + raw


def metadata(key, kind, value):
    return string(key) + u32(kind) + value


def tensor(name="weight", dimensions=(8,), kind=0, offset=0):
    return (string(name) + u32(len(dimensions))
            + b"".join(u64(dim) for dim in dimensions) + u32(kind) + u64(offset))


def gguf(entries=(), tensors=None, payload=b"\0" * 32, alignment=32):
    if tensors is None:
        tensors = [tensor()]
    header = (b"GGUF" + u32(3) + u64(len(tensors)) + u64(len(entries))
              + b"".join(entries) + b"".join(tensors))
    return header + b"\0" * (-len(header) % alignment) + payload


def save(tmp_path, content, name="synthetic.gguf"):
    path = tmp_path / name
    path.write_bytes(content)
    return path


def test_allowlist_ranges_streaming_hash_and_privacy(tmp_path, monkeypatch):
    secret = "SYNTHETIC_DO_NOT_REPORT_TEMPLATE_OR_TOKENIZER_TEXT"
    entries = [
        metadata("general.architecture", 8, string("qwen3vl")),
        metadata("general.name", 8, string("Qwen3-Vl-8B-Instruct")),
        metadata("general.size_label", 8, string("8B")),
        metadata("general.quantized_by", 8, string("Unsloth")),
        metadata("general.license", 8, string("apache-2.0")),
        metadata("general.base_model.0.repo_url", 8,
                 string("https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct")),
        metadata("general.file_type", 4, u32(17)),
        metadata("general.revision", 8, string(secret)),
        metadata("tokenizer.chat_template", 8, string(secret)),
        metadata("private.arbitrary", 8, string(secret)),
        metadata("tokenizer.ggml.tokens", 9, u32(8) + u64(3)
                 + string(secret) + string("<|image_pad|>") + string("<|video_pad|>")),
    ]
    raw = gguf(entries)
    path = save(tmp_path, raw)
    monkeypatch.setattr(candidate, "HASH_CHUNK_BYTES", 7)
    result = candidate.inspect_gguf(path)
    assert result["sha256"] == hashlib.sha256(raw).hexdigest()
    assert result["bytes"] == len(raw)
    assert result["gguf_version"] == 3
    assert result["metadata"]["general.architecture"] == "qwen3vl"
    assert result["metadata"]["general.file_type"] == 17
    assert set(result["metadata"]) <= candidate.STRING_KEYS | candidate.INTEGER_KEYS
    assert result["chat_template_present"] is True
    assert result["visual_token_markers"] == {"image": True, "video": True}
    assert result["tensor_payload_bytes"] == 32
    assert result["unreferenced_data_bytes"] == 0
    assert result["tensor_data_offset"] % 32 == 0
    assert result["revision"] == "unknown"
    assert result["tensor_contents_validated"] is False
    assert result["publisher_checksum_verified"] is False
    assert secret not in json.dumps(result)
    assert "weight" not in json.dumps(result)


@pytest.mark.parametrize("raw,error", [
    (b"NOPE" + b"\0" * 20, "invalid_gguf_magic"),
    (b"GGUF" + u32(2) + b"\0" * 16, "unsupported_gguf_version"),
    (b"GGUF" + u32(99) + b"\0" * 16, "unsupported_gguf_version"),
    (b"GGUF" + u32(3) + u64(100_001) + u64(0), "header_count_limit"),
    (b"GGUF" + u32(3) + u64(1) + u64(16_385), "header_count_limit"),
    (gguf(tensors=[]), "header_count_limit"),
    (gguf([metadata("unknown", 99, b"")]), "unknown_metadata_type"),
    (gguf([metadata("unknown", 9, u32(99) + u64(0))]), "unknown_or_nested_array_type"),
    (gguf([metadata("unknown", 9, u32(9) + u64(0))]), "unknown_or_nested_array_type"),
    (gguf([metadata("unknown", 9, u32(0) + u64(1_000_001))]), "array_limit"),
    (gguf([metadata("unknown", 8, u64(candidate.MAX_STRING_BYTES + 1))]), "string_limit"),
    (gguf([metadata("general.name", 8, u64(4097))]), "string_limit"),
    (gguf([metadata("general.name", 4, u32(1))]), "invalid_allowlisted_metadata_type"),
    (gguf([metadata("general.alignment", 7, b"\1")]), "invalid_allowlisted_metadata_type"),
    (gguf([metadata("general.alignment", 4, u32(3))]), "invalid_tensor_alignment"),
    (gguf([metadata("general.alignment", 4, u32(0))]), "invalid_tensor_alignment"),
    (gguf([metadata("general.name", 8, u64(1) + b"\xff")]), "invalid_metadata_utf8"),
    (gguf([metadata("general.name", 8, string("a"))] * 2), "duplicate_metadata_key"),
    (gguf(tensors=[tensor(kind=999)]), "unknown_tensor_type"),
    (gguf(tensors=[tensor(dimensions=())]), "invalid_tensor_dimensions"),
    (gguf(tensors=[tensor(dimensions=(1,) * 5)]), "invalid_tensor_dimensions"),
    (gguf(tensors=[tensor(dimensions=(0,))]), "invalid_tensor_dimensions"),
    (gguf(tensors=[tensor(dimensions=(2**32 + 1,))]), "invalid_tensor_dimensions"),
    (gguf(tensors=[tensor(dimensions=(2**32, 2**32))]), "invalid_tensor_size"),
    (gguf(tensors=[tensor(offset=1)]), "invalid_tensor_size_or_alignment"),
    (gguf(tensors=[tensor(offset=2**63)]), "truncated_tensor_payload"),
    (gguf(tensors=[tensor(dimensions=(31,), kind=2)]), "invalid_quantized_row_size"),
    (gguf(tensors=[tensor(), tensor()]), "invalid_or_duplicate_tensor_name"),
    (gguf(tensors=[tensor(name="")]), "invalid_or_duplicate_tensor_name"),
    (gguf(tensors=[tensor(), tensor(name="other")]), "overlapping_tensor_ranges"),
    (gguf(payload=b"\0" * 31), "truncated_tensor_payload"),
])
def test_rejects_invalid_or_unbounded_headers(tmp_path, raw, error):
    with pytest.raises(candidate.PreflightError, match=error):
        candidate.inspect_gguf(save(tmp_path, raw))


def test_every_truncated_prefix_is_rejected(tmp_path):
    raw = gguf([metadata("general.name", 8, string("synthetic"))])
    path = tmp_path / "truncated.gguf"
    for end in range(len(raw)):
        path.write_bytes(raw[:end])
        with pytest.raises(candidate.PreflightError):
            candidate.inspect_gguf(path)


def test_bounds_apply_to_skipped_strings_arrays_and_header(tmp_path, monkeypatch):
    path = save(tmp_path, gguf([metadata("ignored", 8, u64(4096))]))
    with pytest.raises(candidate.PreflightError, match="truncated"):
        candidate.inspect_gguf(path)
    path.write_bytes(gguf([metadata("ignored", 9, u32(4) + u64(999))]))
    with pytest.raises(candidate.PreflightError, match="truncated"):
        candidate.inspect_gguf(path)
    monkeypatch.setattr(candidate, "MAX_HEADER_BYTES", 32)
    path.write_bytes(gguf())
    with pytest.raises(candidate.PreflightError, match="header_limit"):
        candidate.inspect_gguf(path)


def test_total_array_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(candidate, "MAX_TOTAL_ITEMS", 3)
    entry = lambda key: metadata(key, 9, u32(0) + u64(2) + b"\0\0")
    path = save(tmp_path, gguf([entry("one"), entry("two")]))
    with pytest.raises(candidate.PreflightError, match="array_limit"):
        candidate.inspect_gguf(path)


def test_file_size_budget_and_nonregular_missing_paths(tmp_path, monkeypatch):
    with pytest.raises(candidate.PreflightError, match="path_missing"):
        candidate.inspect_gguf(tmp_path / "missing.gguf")
    with pytest.raises(candidate.PreflightError, match="file_io_error|not_a_regular_file"):
        candidate.inspect_gguf(tmp_path)
    path = save(tmp_path, gguf())
    monkeypatch.setattr(candidate, "MAX_FILE_BYTES", 16)
    with pytest.raises(candidate.PreflightError, match="file_limit"):
        candidate.inspect_gguf(path)


def test_quantized_ranges_and_custom_alignment(tmp_path):
    raw = gguf([metadata("general.alignment", 4, u32(64))],
               tensors=[tensor(dimensions=(256, 2), kind=13),
                        tensor(name="norm", dimensions=(8,), kind=1, offset=384)],
               payload=b"\0" * 400, alignment=64)
    result = candidate.inspect_gguf(save(tmp_path, raw))
    assert result["tensor_count"] == 2
    assert result["tensor_types"] == {"1": 1, "13": 1}
    assert result["tensor_payload_bytes"] == 352 + 16
    assert result["unreferenced_data_bytes"] == 32
    assert result["tensor_data_offset"] % 64 == 0


def test_report_never_promotes_to_runtime_or_adapter_compatibility(tmp_path):
    path = save(tmp_path, gguf())
    report = candidate.build_report(path)
    assert report["status"] == "metadata_only_preflight"
    assert report["runtime_validation"] == "not_run"
    assert report["visual_pilot"] == "blocked"
    assert "projector_missing_visual_pilot_blocked" in report["blockers"]
    assert report["transformers_peft_directly_loadable"] is False
    assert report["attach_existing_2b_adapter"] is False
    assert report["live_baseline_modified"] is False
    supplied = candidate.build_report(path, path)
    assert supplied["visual_pilot"] == "blocked"
    assert "projector_model_pair_compatibility_unverified" in supplied["blockers"]
    failed = candidate.build_report(path, tmp_path / "missing")
    assert failed["status"] == "preflight_failed"
    assert failed["projector"]["error"] == "path_missing"


def test_cli_new_output_only_and_failure_report(tmp_path, capsys):
    model = save(tmp_path, gguf())
    output = tmp_path / "report.json"
    assert candidate.main(["--model", str(model), "--output", str(output)]) == 0
    saved = output.read_bytes()
    assert json.loads(saved)["visual_pilot"] == "blocked"
    with pytest.raises(SystemExit) as exc:
        candidate.main(["--model", str(model), "--output", str(output)])
    assert exc.value.code == 2
    assert output.read_bytes() == saved
    with pytest.raises(SystemExit):
        candidate.main(["--model", str(model), "--output", str(model)])
    assert model.read_bytes() == gguf()
    failed = tmp_path / "failure.json"
    assert candidate.main(["--model", str(tmp_path / "missing"), "--output", str(failed)]) == 2
    report = json.loads(failed.read_text())
    assert report["model"] == {"status": "failed", "error": "path_missing"}
    assert report["visual_pilot"] == "blocked"
    assert "preflight_failed" in capsys.readouterr().out
