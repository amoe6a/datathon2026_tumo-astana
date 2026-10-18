"""Local, stdlib-only GGUF preflight; never loads tensors or changes the release.

Only allowlisted metadata and aggregate structural checks leave this parser.
SHA-256 streams the file, but is not a publisher checksum verification. Supported
GGML layouts are deliberately explicit: unknown layouts fail closed.
"""
import argparse
import hashlib
import json
import math
import os
import stat
import struct
from collections import Counter
from pathlib import Path


MAX_FILE_BYTES = 64 * 1024**3
MAX_HEADER_BYTES = 64 * 1024**2
MAX_STRING_BYTES = 16 * 1024**2
MAX_ARRAY_ITEMS = 1_000_000
MAX_TOTAL_ITEMS = 2_000_000
MAX_METADATA = 16_384
MAX_TENSORS = 100_000
HASH_CHUNK_BYTES = 4 * 1024**2

STRING_KEYS = frozenset({
    "general.architecture", "general.name", "general.size_label",
    "general.quantized_by", "general.license", "general.base_model.0.repo_url",
})
INTEGER_KEYS = frozenset({
    "general.file_type", "general.quantization_version", "general.alignment",
})
# GGUF value type -> struct layout (strings and arrays are handled separately).
SCALARS = {
    0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f",
    7: "?", 10: "Q", 11: "q", 12: "d",
}
# GGML tensor type -> (elements per block, bytes per block).
TENSOR_LAYOUTS = {
    0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20),
    6: (32, 22), 7: (32, 24), 8: (32, 34), 9: (32, 40),
    10: (256, 84), 11: (256, 110), 12: (256, 144),
    13: (256, 176), 14: (256, 210), 15: (256, 292),
    16: (256, 66), 17: (256, 74), 18: (256, 98), 19: (256, 50),
    20: (32, 18), 21: (256, 110), 22: (256, 82), 23: (256, 136),
    24: (1, 1), 25: (1, 2), 26: (1, 4), 27: (1, 8),
    28: (1, 8), 29: (256, 56), 30: (1, 2),
}


class PreflightError(ValueError):
    """A safe diagnostic code, never arbitrary input content."""


class _Reader:
    def __init__(self, stream, size):
        self.stream = stream
        self.limit = min(size, MAX_HEADER_BYTES)
        self.items = 0
        self.visual_tokens = {"image": False, "video": False}

    def require(self, count):
        if count < 0 or self.stream.tell() + count > self.limit:
            raise PreflightError("truncated_or_header_limit_exceeded")

    def read(self, count):
        self.require(count)
        data = self.stream.read(count)
        if len(data) != count:
            raise PreflightError("truncated_header")
        return data

    def skip(self, count):
        self.require(count)
        self.stream.seek(count, os.SEEK_CUR)

    def number(self, fmt):
        return struct.unpack("<" + fmt, self.read(struct.calcsize("<" + fmt)))[0]

    def string(self, *, retain=False, maximum=MAX_STRING_BYTES, token=False):
        length = self.number("Q")
        if length > maximum:
            raise PreflightError("string_limit_exceeded")
        if retain or (token and length <= 64):
            raw = self.read(length)
            if token:
                if raw == b"<|image_pad|>":
                    self.visual_tokens["image"] = True
                if raw == b"<|video_pad|>":
                    self.visual_tokens["video"] = True
            if retain:
                try:
                    return raw.decode("utf-8", errors="strict")
                except UnicodeDecodeError:
                    raise PreflightError("invalid_metadata_utf8") from None
        else:
            self.skip(length)
        return None

    def value(self, kind, *, retain=False, tokens=False):
        if kind == 8:
            return self.string(retain=retain, maximum=4096 if retain else MAX_STRING_BYTES)
        if kind in SCALARS:
            value = self.number(SCALARS[kind])
            if isinstance(value, float) and not math.isfinite(value):
                raise PreflightError("nonfinite_metadata_number")
            return value if retain else None
        if kind != 9:
            raise PreflightError("unknown_metadata_type")
        element_kind, count = self.number("I"), self.number("Q")
        if count > MAX_ARRAY_ITEMS or self.items + count > MAX_TOTAL_ITEMS:
            raise PreflightError("array_limit_exceeded")
        self.items += count
        if element_kind == 8:
            for _ in range(count):
                self.string(token=tokens)
        elif element_kind in SCALARS:
            self.skip(count * struct.calcsize("<" + SCALARS[element_kind]))
        else:
            # GGUF arrays have scalar/string elements, not nested arrays.
            raise PreflightError("unknown_or_nested_array_type")
        return None


def _parse(stream, size):
    reader = _Reader(stream, size)
    if reader.read(4) != b"GGUF":
        raise PreflightError("invalid_gguf_magic")
    version = reader.number("I")
    if version != 3:
        raise PreflightError("unsupported_gguf_version")
    tensor_count, metadata_count = reader.number("Q"), reader.number("Q")
    if not 0 < tensor_count <= MAX_TENSORS or metadata_count > MAX_METADATA:
        raise PreflightError("header_count_limit_exceeded")
    metadata, seen = {}, set()
    template_present = False
    for _ in range(metadata_count):
        key = reader.string(retain=True, maximum=1024)
        if key in seen:
            raise PreflightError("duplicate_metadata_key")
        seen.add(key)
        kind = reader.number("I")
        retain = key in STRING_KEYS or key in INTEGER_KEYS
        if key in STRING_KEYS and kind != 8:
            raise PreflightError("invalid_allowlisted_metadata_type")
        if key in INTEGER_KEYS and kind not in {0, 1, 2, 3, 4, 5, 10, 11}:
            raise PreflightError("invalid_allowlisted_metadata_type")
        if key == "tokenizer.chat_template":
            template_present = True
        value = reader.value(kind, retain=retain, tokens=key == "tokenizer.ggml.tokens")
        if retain:
            metadata[key] = value
    alignment = metadata.get("general.alignment", 32)
    if alignment < 1 or alignment > 4096 or alignment & (alignment - 1):
        raise PreflightError("invalid_tensor_alignment")
    ranges, names, kinds = [], set(), Counter()
    for _ in range(tensor_count):
        # Names are needed only to reject duplicates; never included in reports.
        name = reader.string(retain=True, maximum=1024)
        if not name or name in names:
            raise PreflightError("invalid_or_duplicate_tensor_name")
        names.add(name)
        dimension_count = reader.number("I")
        if not 1 <= dimension_count <= 4:
            raise PreflightError("invalid_tensor_dimensions")
        dimensions = [reader.number("Q") for _ in range(dimension_count)]
        if any(not 0 < dim <= 2**32 for dim in dimensions):
            raise PreflightError("invalid_tensor_dimensions")
        kind, offset = reader.number("I"), reader.number("Q")
        if kind not in TENSOR_LAYOUTS:
            raise PreflightError("unknown_tensor_type")
        block, width = TENSOR_LAYOUTS[kind]
        if dimensions[0] % block:
            raise PreflightError("invalid_quantized_row_size")
        length = math.prod(dimensions) // block * width
        if length > MAX_FILE_BYTES or offset % alignment:
            raise PreflightError("invalid_tensor_size_or_alignment")
        ranges.append((offset, offset + length))
        kinds[kind] += 1
    header_end = stream.tell()
    data_start = (header_end + alignment - 1) // alignment * alignment
    if data_start > size:
        raise PreflightError("truncated_tensor_alignment_padding")
    previous_end = 0
    for start, end in sorted(ranges):
        if start < previous_end:
            raise PreflightError("overlapping_tensor_ranges")
        if data_start + end > size:
            raise PreflightError("truncated_tensor_payload")
        previous_end = end
    return {
        "gguf_version": version,
        "metadata": metadata,
        "metadata_entries": metadata_count,
        "chat_template_present": template_present,
        "visual_token_markers": reader.visual_tokens,
        "tensor_count": tensor_count,
        "tensor_types": {str(kind): count for kind, count in sorted(kinds.items())},
        "header_bytes": header_end,
        "tensor_data_offset": data_start,
        "tensor_payload_bytes": sum(end - start for start, end in ranges),
        "unreferenced_data_bytes": size - data_start - sum(end - start for start, end in ranges),
        "structure_status": "header_and_known_tensor_ranges_checked",
        "revision": "unknown",
        "publisher_checksum_verified": False,
        "tensor_contents_validated": False,
    }


def inspect_gguf(path):
    """Parse a regular GGUFv3 file with bounded memory, then stream its SHA-256.

    Reads at most MAX_HEADER_BYTES for structure and MAX_FILE_BYTES for hashing.
    No tensor values are interpreted. File changes during inspection fail closed.
    """
    try:
        fd = os.open(os.fspath(path), os.O_RDONLY | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise PreflightError("not_a_regular_file")
            if before.st_size > MAX_FILE_BYTES:
                raise PreflightError("file_limit_exceeded")
            result = _parse(stream, before.st_size)
            stream.seek(0)
            digest, remaining = hashlib.sha256(), before.st_size
            while remaining:
                chunk = stream.read(min(HASH_CHUNK_BYTES, remaining))
                if not chunk:
                    raise PreflightError("file_changed_during_inspection")
                digest.update(chunk)
                remaining -= len(chunk)
            after = os.fstat(stream.fileno())
            if stream.read(1) or (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                after.st_size, after.st_mtime_ns, after.st_ctime_ns
            ):
                raise PreflightError("file_changed_during_inspection")
            result.update(bytes=before.st_size, sha256=digest.hexdigest())
            return result
    except FileNotFoundError:
        raise PreflightError("path_missing") from None
    except OSError:
        raise PreflightError("file_io_error") from None


def build_report(model, projector=None):
    report = {
        "status": "metadata_only_preflight",
        "runtime_validation": "not_run",
        "live_baseline_modified": False,
        "transformers_peft_directly_loadable": False,
        "attach_existing_2b_adapter": False,
        "visual_pilot": "blocked",
        "blockers": ["compatible_local_gguf_visual_runtime_not_verified"],
        "limitations": [
            "GGUF is not directly loadable by the existing Transformers/PEFT pipeline.",
            "Do not attach the existing 2B adapter to this larger quantized model.",
            "Metadata claims do not establish publisher identity, license verification or revision.",
            "No inference, projector compatibility, quality, latency or GPU-memory validation was run.",
        ],
    }
    for label, path in (("model", model), ("projector", projector)):
        if path is None:
            report[label] = {"status": "not_provided"}
            report["blockers"].append("projector_missing_visual_pilot_blocked")
            continue
        try:
            report[label] = inspect_gguf(path)
        except PreflightError as exc:
            report[label] = {"status": "failed", "error": str(exc)}
            report["blockers"].append(label + "_preflight_failed")
            report["status"] = "preflight_failed"
    if projector is not None:
        report["blockers"].append("projector_model_pair_compatibility_unverified")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--projector", type=Path)
    parser.add_argument("--output", required=True, type=Path, help="New report path; never overwritten")
    args = parser.parse_args(argv)
    # Reserve a NEW file before expensive hashing; do not overwrite inputs/reports.
    try:
        with args.output.open("x", encoding="utf-8") as output:
            report = build_report(args.model, args.projector)
            json.dump(report, output, indent=2, ensure_ascii=True, allow_nan=False)
            output.write("\n")
    except OSError:
        parser.exit(2, "Cannot create/write new output report; existing paths are never overwritten.\n")
    print(json.dumps({"status": report["status"], "visual_pilot": report["visual_pilot"]}))
    return 2 if report["status"] == "preflight_failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
