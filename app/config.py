import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

@dataclass
class Settings:
    api_key: str
    engine: str = "reference"
    model: Path = ROOT / "models/reference.npz"
    runtime: Path = ROOT / "runtime"
    rules: Path = ROOT / "rules/catalog.json"
    timeout: float = 270
    max_upload_mb: int = 64

    @classmethod
    def from_env(cls):
        key = os.getenv("FORTE_API_KEY", "")
        if len(key) < 16:
            raise RuntimeError("Set FORTE_API_KEY to a random secret of at least 16 characters.")
        engine = os.getenv("FORTE_ENGINE", "reference")
        if engine not in {"demo", "reference", "qwen"}:
            raise RuntimeError("FORTE_ENGINE must be demo, reference or qwen")
        return cls(key, engine, Path(os.getenv("FORTE_MODEL", str(ROOT / "models/reference.npz"))),
                   Path(os.getenv("FORTE_RUNTIME", str(ROOT / "runtime"))),
                   Path(os.getenv("FORTE_RULES", str(ROOT / "rules/catalog.json"))),
                   min(270, float(os.getenv("FORTE_JOB_TIMEOUT", "270"))),
                   int(os.getenv("FORTE_MAX_UPLOAD_MB", "64")))
