# Linux release deployment

The dedicated project lives at `/home/angel/forte-review-service`. Ratmir's original development checkouts are preserved. The public URL is **https://fracture-freestyle-unvalued.ngrok-free.dev**; it forwards requests to this laptop rather than a hosted AI model.

## Running services

| User unit | Purpose | Local endpoint |
|---|---|---|
| `forte-jury.service` | Selected Qwen model, authenticated API and UI | `127.0.0.1:8000` |
| `forte-ngrok.service` | HTTPS forwarding, traffic inspection disabled | Existing ngrok domain |
| `forte-awake.service` | Inhibit sleep during the judging window | None |
| `forte-demo.service` | Explicitly synthetic CPU demonstration | `127.0.0.1:8001` |

The former `forte-jury-model.service` is Bonsai and is disabled. Do not start it alongside Qwen. Cloudflare did not connect on this network; the existing ngrok service is the current transport.

The final model configuration is `configs/qwen-release.json`. The API key is preserved at `runtime/forte-jury/access-key.txt`; `runtime/forte-jury/service.env` also contains it. Never print or commit either file. Runtime directories are private (0700), keys/environment files are 0600.

## Start, inspect and recover

As the `angel` user:

```bash
cd /home/angel/forte-review-service
systemctl --user status forte-jury.service forte-ngrok.service forte-awake.service
curl --fail http://127.0.0.1:8000/ready
curl --fail https://fracture-freestyle-unvalued.ngrok-free.dev/ready

# Preserve the current address, model selection, key and completed jobs:
systemctl --user restart forte-jury.service
systemctl --user restart forte-ngrok.service
```

The API and tunnel use `Restart=on-failure`; user lingering is enabled. A restart interrupts active work and marks it failed instead of giving a clean verdict. Completed jobs persist in SQLite. Do not restart during a judge upload unless recovery is necessary.

Model/configuration/catalog changes require restarting the API. Readiness performs a bounded synthetic inference probe; a reachable `/health` alone does not establish model readiness. The service accepts one job at a time and returns 429 with `Retry-After` for overload. It uses bounded child processes, so every real upload includes a model load.

## Environment and reproducibility

The final API uses `.venv-release/bin/python`. This environment was copied from the already tested CUDA packages into its own directory, without the `.pth` reference to Ratmir's original ML environment. Imports and the test suite were checked in that environment; actual inference checks are recorded in the release report. It is a local deployment environment, not a portable virtualenv archive.

The tested Linux stack uses Python 3.14.8, Torch 2.14.0+cu130, Transformers 5.17.0, PEFT 0.21.2, bitsandbytes 0.50.2 and the pinned LM Format Enforcer revision. `requirements-linux-qwen-lock.txt` records the dependency closure used for runtime and tests. `requirements-tested-lock.txt` is the separate Windows CPU test environment.

For a fresh compatible Linux machine, create a virtualenv using Python 3.14, install CUDA Torch/Torchvision from the official cu130 wheel index, then install the Linux lock file. Install the exact LM Format Enforcer Git revision in that file even if a same-version PyPI build is already present. Download the pinned publisher model and copy the selected private adapter before switching offline. Run imports, offline inference and a real MOV before treating a fresh installation as verified. Driver compatibility and wheel availability on another machine have not been tested.

The current RTX laptop can execute CUDA despite an NVML driver/library mismatch. Do not replace drivers or reboot just to repair `nvidia-smi` during judging. CUDA allocation measurements come from PyTorch, not a working NVML per-process monitor.

## Reinstall the owned service

```bash
cd /home/angel/forte-review-service
.venv-release/bin/python -m scripts.install_service \
  --name forte-jury --engine qwen_rag --port 8000 \
  --config configs/qwen-release.json \
  --python /home/angel/forte-review-service/.venv-release/bin/python
systemctl --user daemon-reload
systemctl --user restart forte-jury.service
```

The installer preserves the existing key, generates the matching rule catalog, disables keyless testing (`FORTE_LOCAL_TEST=0`), and only updates its own named units. It does not create a new public tunnel or account.

## Rollback

The earlier verified 2B full-clip configuration is `configs/qwen-rag-2b-bf16.json`, with the four-epoch scene adapter at `models/forte-qwen-2b-scene-epoch4`. Before changing a deployment, preserve the current configuration, unit and private environment file in a private runtime backup. Create a separate rollback config with that adapter path; reinstall the owned service with it and run readiness plus an authenticated MOV check. Never reset the runtime database or regenerate the judge key as part of rollback.

Private training checkpoints, reports and prior deployment snapshots remain under `runtime/qwen-training-20261017`. A pre-release Git stash, when created during synchronization, preserves earlier copied source changes; do not pop it blindly over the release.

## Availability and data

Keep both Ratmir's laptop and the Windows hotspot laptop plugged in, awake and connected until at least **18 October 2026, 17:00 UTC+05**. The Windows keep-awake process and Linux sleep inhibitor are task-specific. Avoid changing the network or running GPU experiments during judging. Power, connectivity and provider quota remain external dependencies; no uninterrupted-uptime guarantee is claimed.

See [judge handoff](judges-handoff.md), [Qwen training and limitations](qwen-training-deployment.md), and the README's final release report. Synthetic demonstrations are distinct from measured real-video detection. Event data and detailed predictions stay on the laptop; do not publish them or retain judge recordings beyond the organizer's permission.
