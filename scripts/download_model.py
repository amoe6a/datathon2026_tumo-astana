"""Setup-only model download with pinned revision and local artifact hashes."""
import argparse
import json
from pathlib import Path
import re
from huggingface_hub import snapshot_download
from ml.manifest import sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="models/qwen3-vl-2b")
    parser.add_argument("--revision", default="78448d793a7eb2f7a987a1da76d464384aa1becd")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        parser.error("revision must be a full immutable commit SHA")
    snapshot_download("Qwen/Qwen3-VL-2B-Instruct", revision=args.revision, local_dir=args.output,
                      allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "LICENSE", "README.md"])
    output = Path(args.output)
    files = [p for p in sorted(output.rglob("*")) if p.is_file()
             and ".cache" not in p.relative_to(output).parts and p.name != "setup-manifest.json"]
    manifest = {"repo_id": "Qwen/Qwen3-VL-2B-Instruct", "revision": args.revision,
                "license_reference": "README.md (publisher model card); retain LICENSE if supplied",
                "files": [{"path": p.relative_to(output).as_posix(), "bytes": p.stat().st_size,
                           "sha256": sha256(p)} for p in files]}
    (output / "setup-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"downloaded_files": len(files), "revision": args.revision,
                      "total_bytes": sum(f["bytes"] for f in manifest["files"]), "hashes_saved": True}))


if __name__ == "__main__":
    main()
