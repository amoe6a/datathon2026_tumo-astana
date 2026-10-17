import argparse
import base64
import os
import time
from pathlib import Path
import httpx

p=argparse.ArgumentParser();p.add_argument("video");p.add_argument("--url",default="http://127.0.0.1:8000");a=p.parse_args()
key=os.environ.get("FORTE_API_KEY","")
if not key:raise SystemExit("Set FORTE_API_KEY in your shell")
start=time.monotonic()
with httpx.Client(base_url=a.url,timeout=60,trust_env=False) as client:
    with open(a.video,"rb") as f:
        r=client.post('/api/v1/analyze',headers={"X-API-Key":key},files={"file":(Path(a.video).name,f,"video/quicktime")})
    r.raise_for_status();jid=r.json()["job_id"]
    while time.monotonic()-start<300:
        r=client.get(f'/api/v1/jobs/{jid}',headers={"X-API-Key":key});r.raise_for_status();result=r.json()
        if result['status']=='failed':raise SystemExit("FAIL: "+result['error'])
        if result['status']=='done':
            assert result['verdict'] in ['violation','no_violation']
            assert (result['verdict']=='violation')==bool(result['violations'])
            for v in result['violations']:
                assert 1<=len(v['evidence_frames'])<=3
                for frame in v['evidence_frames']:
                    assert base64.b64decode(frame['image_base64'],validate=True).startswith(b'\xff\xd8')
            print(f"PASS: status=done, events={len(result['violations'])}, end-to-end seconds={time.monotonic()-start:.2f}")
            break
        time.sleep(1)
    else:raise SystemExit('FAIL: 300-second jury budget exceeded')
