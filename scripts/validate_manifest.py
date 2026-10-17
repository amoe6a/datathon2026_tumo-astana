"""Print aggregate checks only. Draft manifests must not be used for training."""
import argparse
import json
from ml.manifest import ManifestError, validate_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("--rules", default="rules/catalog.json")
    parser.add_argument("--purpose", choices=["train", "evaluate"], default="train")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    try:
        _, report = validate_manifest(args.manifest, args.rules, purpose=args.purpose, demo=args.demo)
    except ManifestError as error:
        parser.exit(1, f"Manifest rejected: {error}\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
