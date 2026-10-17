"""Setup-only pinned Bonsai download, or offline verification of existing files."""
import argparse
import json
from pathlib import Path
from ml.manifest import sha256

REPO = 'prism-ml/Ternary-Bonsai-2-27B-gguf'
REVISION = 'b072e1d3b35a0a630cece372c2127528e0994386'
FILES = {
    'Ternary-Bonsai-2-27B-PTQ1_0.gguf': (5946648928, '53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3'),
    'Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf': (629246976, '6807ede61d570bb86ba34b756a0fa109edc33668604de867c6ea6d8f1d631903'),
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', default='models/bonsai')
    p.add_argument('--verify-only', action='store_true')
    a = p.parse_args(); root = Path(a.output); root.mkdir(parents=True, exist_ok=True)
    if not a.verify_only:
        from huggingface_hub import hf_hub_download
        for name in FILES:
            hf_hub_download(REPO, name, revision=REVISION, local_dir=str(root))
    artifacts = []
    for name, (size, expected) in FILES.items():
        path = root / name
        if not path.is_file() or path.stat().st_size != size or sha256(path) != expected:
            raise ValueError('Pinned Bonsai artifact is missing or its checksum does not match')
        artifacts.append({'path': name, 'bytes': size, 'sha256': expected})
    manifest = {'repo': REPO, 'revision': REVISION, 'files': artifacts, 'publisher_checksums_verified': True}
    (root / 'setup-manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({'verified_artifacts': len(artifacts), 'revision': REVISION}))


if __name__ == '__main__':
    main()
