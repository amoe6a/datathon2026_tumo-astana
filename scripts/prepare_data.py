"""Import organizer XLSX + video ZIP locally; never infer semantic annotations.

Only aggregate summaries go to stdout. Source labels/times remain in ignored
data/. Uses stdlib OOXML and ffprobe so shared dependencies remain unchanged.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
import zipfile

from ml.manifest import sha256

NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def read_xlsx(path):
    with zipfile.ZipFile(path) as archive:
        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            strings = ["".join(e.itertext()) for e in ET.fromstring(archive.read("xl/sharedStrings.xml")).findall("s:si", NS)]
        root = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        result = []
        for row in root.findall("s:sheetData/s:row", NS):
            cells = {}
            for cell in row:
                column = re.sub(r"\d", "", cell.attrib["r"])
                value = cell.find("s:v", NS)
                inline = cell.find("s:is", NS)
                text = value.text if value is not None else "".join(inline.itertext()) if inline is not None else ""
                cells[column] = strings[int(text)] if cell.get("t") == "s" else text
            if any(cells.values()):
                result.append(cells)
        return result


def parse_time(value):
    value = str(value).strip().lower()
    if not value:
        return None
    if re.fullmatch(r"\d+(?:[.,]\d+)?", value):
        return float(value.replace(",", "."))
    if re.fullmatch(r"\d+:\d{1,2}(?::\d{1,2}(?:\.\d+)?)?", value):
        parts = list(map(float, value.split(":")))
        if any(v >= 60 for v in parts[1:]):
            raise ValueError("Invalid timestamp component")
        return sum(v * 60 ** i for i, v in enumerate(reversed(parts)))
    match = re.fullmatch(r"(?:(\d+)\s*мин(?:ута|уты|ут)?\s*)?(\d+(?:[.,]\d+)?)\s*сек(?:унда|унды|унд)?", value)
    if match:
        return int(match[1] or 0) * 60 + float(match[2].replace(",", "."))
    raise ValueError("Unsupported source timestamp format; review locally")


def inventory(video):
    completed = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height,avg_frame_rate,duration:format=duration", "-of", "json", str(video)],
        capture_output=True, text=True, timeout=30)
    if completed.returncode:
        raise ValueError("Video metadata probing failed")
    metadata = json.loads(completed.stdout)
    stream = metadata["streams"][0]
    return {"duration_sec": float(stream.get("duration", metadata["format"].get("duration"))),
            "width": stream["width"], "height": stream["height"],
            "codec": stream["codec_name"], "avg_frame_rate": stream["avg_frame_rate"],
            "sha256": sha256(video), "bytes": video.stat().st_size}


def prepare(archive_path, output):
    output = Path(output)
    # Never overwrite human annotations or a previously imported dataset.
    output.mkdir(parents=True, exist_ok=False)
    raw = output / "raw"
    raw.mkdir()
    with zipfile.ZipFile(archive_path) as archive:
        members = [m for m in archive.infolist() if not m.is_dir()]
        names = set()
        for member in members:
            name = Path(member.filename).name
            if name in names or "\\" in name:
                raise ValueError("Ambiguous archive filenames")
            names.add(name)
            if Path(name).suffix.lower() not in {".mov", ".mp4", ".xlsx"}:
                raise ValueError("Unexpected archive member type")
            with archive.open(member) as source, (raw / name).open("xb") as target:
                shutil.copyfileobj(source, target)
    sheets = list(raw.glob("*.xlsx"))
    if len(sheets) != 1:
        raise ValueError("Exactly one XLSX annotation sheet required")
    rows = read_xlsx(sheets[0])
    if not rows or [rows[0].get(k, "").strip().lower() for k in "ABC"] != ["наименование файла", "таймстемп", "нарушение / без нарушения"]:
        raise ValueError("Unexpected organizer sheet schema")
    videos = {}
    for video in raw.iterdir():
        if video.suffix.lower() in {".mov", ".mp4"}:
            if video.stem.casefold() in videos:
                raise ValueError("Ambiguous video stems")
            videos[video.stem.casefold()] = video
    records, seen = [], set()
    for row in rows[1:]:
        stem = Path(row.get("A", "").strip()).stem.casefold()
        if stem not in videos or stem in seen:
            raise ValueError("Missing or repeated organizer-to-video mapping")
        seen.add(stem)
        label_raw, time_raw = row.get("C", ""), row.get("B", "")
        label = {"нарушение": 1, "без нарушения": 0}.get(label_raw.strip().lower())
        if label is None:
            raise ValueError("Unrecognized organizer binary label")
        timestamp = parse_time(time_raw)
        metadata = inventory(videos[stem])
        if timestamp is not None and not 0 <= timestamp <= metadata["duration_sec"]:
            raise ValueError("Organizer timestamp outside clip")
        records.append({"path": videos[stem].relative_to(output).as_posix(), **metadata,
                        "source": {"binary_label": label, "timestamp_sec": timestamp,
                                   "filename_raw": row["A"], "label_raw": label_raw, "timestamp_raw": time_raw},
                        "group": None, "suggested_exact_group": "sha256_" + metadata["sha256"],
                        "group_reviewed": False, "split": None,
                        "segments": [], "events": [], "events_reviewed": False,
                        "review": {"assignee": f"labeler_{(len(records) % 3) + 1}", "uncertain": None, "notes": ""}})
    if seen != set(videos):
        raise ValueError("Some videos have no organizer annotation")
    draft = {"schema_version": 1, "state": "draft",
             "note": "Human grouping, split and explicit intervals required. Source timestamps are not event intervals. No implicit normal frames.",
             "videos": records}
    (output / "manifest.draft.json").write_text(json.dumps(draft, ensure_ascii=False, indent=2), encoding="utf-8")
    counts = Counter(r["sha256"] for r in records)
    summary = {"videos": len(records), "positive": sum(r["source"]["binary_label"] for r in records),
               "negative": sum(not r["source"]["binary_label"] for r in records),
               "source_timestamps": sum(r["source"]["timestamp_sec"] is not None for r in records),
               "total_duration_sec": round(sum(r["duration_sec"] for r in records), 3),
               "duration_range_sec": [round(fn(r["duration_sec"] for r in records), 3) for fn in (min, max)],
               "exact_duplicate_copies": sum(v - 1 for v in counts.values()),
               "assignments": dict(Counter(r["review"]["assignee"] for r in records)),
               "ready_for_training": False}
    (output / "inventory.summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive")
    parser.add_argument("--output", default="data/real")
    args = parser.parse_args()
    print(json.dumps(prepare(args.archive, args.output), indent=2))
