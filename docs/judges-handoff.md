# Judge API and operating instructions

Public UI and API base: **https://fracture-freestyle-unvalued.ngrok-free.dev**

This address forwards HTTPS requests to the team's local RTX laptop. Video inference does not call a cloud AI service. The selected release, measured results and remaining limits are recorded in the release report linked from the README.

The first browser visit can show ngrok's own “Visit Site” notice; select it to reach the interface. This notice was observed during the public UI check. The tested API clients reached JSON responses directly. It is separate from a browser certificate warning, which should not be bypassed.

## Access

The team supplies one private access key to the judges. Send it in the `X-API-Key` header on both endpoints. The same key can be entered in the UI; it is held for the page session and is not embedded in the site or committed to Git.

On the deployed laptop the key is at `/home/angel/forte-review-service/runtime/forte-jury/access-key.txt`. Send its value through the organizer's private submission channel. The public repository intentionally contains no key.

```bash
# Run on Ratmir's laptop. This loads the key without printing it.
FORTE_API_KEY="$(cat /home/angel/forte-review-service/runtime/forte-jury/access-key.txt)"
curl --fail-with-body -X POST \
  https://fracture-freestyle-unvalued.ngrok-free.dev/api/v1/analyze \
  -H "X-API-Key: $FORTE_API_KEY" -F 'file=@/absolute/path/to/video.mov'

# Replace JOB_ID with the ID in the HTTP 202 upload response.
curl --fail-with-body \
  https://fracture-freestyle-unvalued.ngrok-free.dev/api/v1/jobs/JOB_ID \
  -H "X-API-Key: $FORTE_API_KEY"
unset FORTE_API_KEY
```

## Contract

- `POST /api/v1/analyze`: multipart field `file`, accepts MOV; returns HTTP 202 with `job_id` and `status: processing`.
- `GET /api/v1/jobs/{job_id}`: `processing`, `done`, or `failed`.
- A completed result has `verdict: violation` or `no_violation`, and `violations` (empty for a clean result).
- Each event includes `type`, `timecode_sec`, `description`, and `evidence_frames`: one to three JPEGs with `timestamp_sec` and `image_base64`.
- Additional rule/source metadata supports human review without changing required fields.
- Missing/wrong keys receive HTTP 401. Upload limits are 64 MiB and 180 seconds. One active job is supported; concurrent uploads receive HTTP 429 and `Retry-After`.
- Poll approximately every three seconds. A failed job is an explicit processing failure, never a clean verdict. End-to-end processing must meet the case's five-minute limit, including upload.

`GET /health` checks the HTTP service; **`GET /ready` must return HTTP 200 with `ready: true`** before submission. A live tunnel alone is insufficient.

## Before and during judging

Keep Ratmir's laptop and the Windows hotspot laptop plugged in, awake and connected through **18 October 2026, 17:00 Asia/Qyzylorda (UTC+05)**. Do not change Wi-Fi, close the hotspot, run GPU training, or switch models during judging.

On Ratmir's laptop, as `angel`:

```bash
systemctl --user status forte-jury.service forte-ngrok.service forte-awake.service
curl --fail http://127.0.0.1:8000/ready
curl --fail https://fracture-freestyle-unvalued.ngrok-free.dev/ready

# Recover the existing service/tunnel without creating a new URL or key:
systemctl --user restart forte-jury.service
systemctl --user restart forte-ngrok.service
```

Restarting the API interrupts an active upload/analysis, which is marked failed; wait for active work to finish when possible. Completed results and the key survive restart. Never start the old `forte-jury-model.service`: it is the retired Bonsai process and competes for GPU memory.

The units restart on failure and user lingering is enabled. Keep-awake settings cannot prevent power failure, a disconnected hotspot, provider quota exhaustion, or an ISP outage. The ngrok account's remaining quota is not established by a successful smoke test; check it in the account dashboard before submission.

No automatic submission to the organizers has been made. The team must deliver the URL and private key through the official submission channel and confirm receipt.

## Quality boundaries

This is a review aid, not a legal or disciplinary determination. A syntactically valid response does not establish that the action, actor role, timestamp or rule interpretation is correct. The validation/test sets are small, many validation configurations were tried, and a genuinely different camera angle has not been independently validated. Do not present synthetic examples as real model accuracy.

All competition footage, annotations, detailed model descriptions and private evidence remain on the local host. Do not publish judge recordings or use them for further training. Follow the organizer's post-event retention requirements.
