"""Wait for an already configured loopback model server before API startup."""
import argparse
import json
from pathlib import Path
import time
import httpx
from ml.bonsai import checked_endpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--timeout', type=float, default=60)
    args = parser.parse_args()
    endpoint = checked_endpoint(json.loads(Path(args.config).read_text(encoding='utf-8'))['endpoint'])
    deadline = time.monotonic() + args.timeout
    with httpx.Client(timeout=2, trust_env=False, follow_redirects=False) as client:
        while time.monotonic() < deadline:
            try:
                if client.get(endpoint + '/health').status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(.25)
    raise SystemExit('Local model server did not become healthy before the startup deadline')


if __name__ == '__main__':
    main()
