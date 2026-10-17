import argparse
import json
import os
import time
from pathlib import Path

def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    for attempt in range(5):
        try:
            temp.replace(path)
            return
        except PermissionError:
            # A Windows reader may briefly prevent an atomic replacement.
            # Persistent errors still fail the worker; never lose a final result silently.
            if attempt == 4:
                raise
            time.sleep(0.01 * 2**attempt)

def main():
    p=argparse.ArgumentParser()
    for name in ["input", "model", "engine", "rules", "output", "progress"]:
        p.add_argument("--"+name, required=True)
    a=p.parse_args()
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    from ml.engine import analyze
    try:
        result = analyze(a.input, a.model, a.engine, a.rules,
                         lambda percent, stage: atomic_json(a.progress, {"progress":percent, "stage":stage}))
        atomic_json(a.output, {"ok":True, "result":result})
    except Exception as e:
        # No video descriptions or model output in logs.
        safe = str(e) if isinstance(e, ValueError) else f"{type(e).__name__}: local model or decoder failed; check installation and GPU memory"
        atomic_json(a.output, {"ok":False, "error":safe[:500]})

if __name__ == "__main__":
    main()
