import os
import math
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
    readiness_timeout: float = 90

    def __post_init__(self):
        if not math.isfinite(self.timeout) or not 0 < self.timeout <= 270:
            raise ValueError('Processing timeout must be between 0 and 270 seconds')
        if not math.isfinite(self.readiness_timeout) or not 0 < self.readiness_timeout <= 120:
            raise ValueError('Readiness timeout must be between 0 and 120 seconds')
        if not isinstance(self.max_upload_mb, int) or self.max_upload_mb <= 0:
            raise ValueError('Upload limit must be a positive number of MiB')

    @classmethod
    def from_env(cls):
        key = os.getenv("FORTE_API_KEY", "")
        if len(key) < 16:
            raise RuntimeError("Set FORTE_API_KEY to a random secret of at least 16 characters.")
        engine = os.getenv("FORTE_ENGINE", "reference")
        if engine not in {"demo", "reference", "qwen", "qwen_rag"}:
            raise RuntimeError("FORTE_ENGINE must be demo, reference, qwen or qwen_rag")
        return cls(key, engine, Path(os.getenv("FORTE_MODEL", str(ROOT / "models/reference.npz"))),
                   Path(os.getenv("FORTE_RUNTIME", str(ROOT / "runtime"))),
                   Path(os.getenv("FORTE_RULES", str(ROOT / "rules/catalog.json"))),
                   min(270, float(os.getenv("FORTE_JOB_TIMEOUT", "270"))),
                   int(os.getenv("FORTE_MAX_UPLOAD_MB", "64")),
                   min(120, float(os.getenv("FORTE_READINESS_TIMEOUT", "90"))))
