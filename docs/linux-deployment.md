# Ratmir's Linux deployment

The backend is installed in a separate checkout at `/home/angel/forte-review-service`, on branch `codex/ratmir-deployment`. Ratmir's original main and ML checkouts are preserved. The deployed `.venv` and `models/bonsai` are links to his existing ML environment and model directory; keep those source directories in place.

## Services and addresses

These addresses refer to **Ratmir's laptop**. They are bound to loopback and are not public URLs.

| Unit | Purpose | Local address |
|---|---|---|
| `forte-demo.service` | Explicitly labelled synthetic demonstration | `http://127.0.0.1:8001` |
| `forte-jury-model.service` | Native Bonsai inference, loopback only | `http://127.0.0.1:18080` |
| `forte-jury.service` | Real-model API and UI | `http://127.0.0.1:8000` |

All are systemd **user** services belonging to `angel`. The existing user configuration has `Linger=yes`, and the services are enabled for the user manager. They use `Restart=on-failure`, a three-second restart delay and control-group cleanup. No system driver or firewall settings were changed.

```bash
# On Ratmir's laptop, as angel:
systemctl --user status forte-demo.service forte-jury-model.service forte-jury.service
systemctl --user restart forte-demo.service

# Restart both real-model processes, in order:
systemctl --user stop forte-jury.service
systemctl --user restart forte-jury-model.service
systemctl --user start forte-jury.service
curl -f http://127.0.0.1:8000/ready

# Stop real inference to release GPU memory:
systemctl --user stop forte-jury.service
systemctl --user stop forte-jury-model.service

# Start it again:
systemctl --user start forte-jury-model.service forte-jury.service
```

The API waits for the native model's health endpoint before starting its own bounded synthetic inference probe. Its `/ready` endpoint also checks the native server's availability before accepting new uploads; it returns 503 during an outage and recovers once the same model server is healthy. A changed model configuration or rule catalog requires restarting the API.

## Credentials and logs

The two API instances have independent keys. No account or per-judge registration is required: give the judges the real service's team key privately after public access is verified.

- Demo key: `/home/angel/forte-review-service/runtime/forte-demo/access-key.txt`
- Real API key: `/home/angel/forte-review-service/runtime/forte-jury/access-key.txt`
- Environment files: `service.env` in each runtime directory; these also contain the key.
- API logs: `runtime/forte-demo/api.log` and `runtime/forte-jury/api.log`.
- Native log: `runtime/forte-jury/native.log`. Native request logging is disabled.
- Generated unit files: `~/.config/systemd/user/forte-*.service`.

Runtime directories use mode 0700; keys and environment files use 0600. They are ignored by Git. The installer prints locations and service metadata, never key values. Do not include keys or environment files in a report or screenshot.

## Model and environment

The API runs in the existing Python 3.11.15 core environment. Ratmir's separate CUDA environment was checked with a real tensor calculation: Python 3.14.8, torch 2.14.0+cu130, RTX 4060 Laptop GPU. Bonsai itself uses the native CUDA runtime, independently of PyTorch.

Both Bonsai GGUF files were checked against the publisher's pinned SHA-256 values. The configuration is `configs/bonsai.json`; model revision is `b072e1d3b35a0a630cece372c2127528e0994386`, and the native source revision recorded by Ratmir is `e3de715222b2667c54a267441159a5729f11155e`. The native binary is reused from his ML checkout's `runtime/bonsai-setup/build/bin/llama-server`.

`nvidia-smi` currently reports an NVML driver/library mismatch: loaded kernel module 615.71.09, NVML library 615.78. CUDA execution works. A native per-process VRAM peak could not be measured with NVML during this deployment; no such peak is claimed. Driver replacement/reboot was not attempted during ongoing work.

A separate controlled CUDA memory check found approximately **5.843 GiB** more GPU memory occupied with the native model running than with it stopped (6,273,761,280 bytes). This supports actual GPU offload. It is a pair of global memory snapshots, not a per-process peak. The API was restored and passed `/ready` afterward.

## Executed acceptance checks

The tested application revision is `c52b80e41d7e9b8606b6f57ef59be3cfe579a904`; later documentation/report changes preserve that application code. The combined code passed 87 Python/API tests on Windows and again on Linux; the final Linux run took 25.88 seconds. All 11 frontend checks and JavaScript syntax passed on Linux. The existing Starlette/HTTPX deprecation warning remains.

| Live check | Result | Upload-to-result seconds |
|---|---|---:|
| Synthetic normal | done / no_violation | 0.521 |
| Synthetic one event | done / violation, 3 verified JPEGs | 0.548 |
| Synthetic two events | done / violation, 6 verified JPEGs | 0.520 |
| Synthetic corrupt file | failed, no verdict | 0.265 |
| Real negative train clip, 15.867 s | done / no_violation | 12.913 |
| Real positive train clip, 14.700 s | done / violation, 3 verified JPEGs | 14.432 |

Three runs on the longest train clip (142.533 seconds) stopped and reloaded both native/API processes. Including startup, complete times were **139.507 / 143.516 / 143.674 seconds**. Upload-to-result times after readiness were **133.716 / 137.242 / 136.254 seconds**. All completed and passed the response checks. These were host-loopback requests; disk cache was retained. The returned verdict is not a correctness claim.

The native process's maximum observed VmHWM across those runs was 6,484,344,832 bytes; the API process maximum was 98,639,872 bytes. These are separate process RSS measurements, not a combined system peak.

Crash tests killed only our dedicated units. Native restart restored availability after an observed 503 outage; a new real train upload then completed. Killing the demo API during an accepted upload produced `failed` after restart, removed the input file, retained earlier completed results and allowed a subsequent upload. The two API keys were confirmed isolated.

Both SSH sessions were then closed. A fresh login 32 seconds later found all three service PIDs unchanged, with both API readiness endpoints returning 200. A host reboot was not performed. Full aggregate evidence is in [the deployment report](../reports/backend-deployment-20261017.json).

## Reinstalling service definitions

Use the existing environment and already verified files. The installer updates only dedicated units carrying its own generated-file header and preserves the existing key.

```bash
cd /home/angel/forte-review-service
.venv/bin/python -m scripts.install_service --name forte-demo --engine demo --port 8001

# Set this to the existing, pinned binary in Ratmir's ML checkout:
NATIVE_BINARY=/absolute/path/to/runtime/bonsai-setup/build/bin/llama-server
.venv/bin/python -m scripts.install_service --name forte-jury --engine bonsai \
  --port 8000 --config configs/bonsai.json --native-binary "$NATIVE_BINARY"
systemd-analyze --user verify ~/.config/systemd/user/forte-demo.service \
  ~/.config/systemd/user/forte-jury.service ~/.config/systemd/user/forte-jury-model.service
systemctl --user daemon-reload
```

The installer does not start services or publish an endpoint. Use the explicit commands above after checking the definitions. To roll back code, stop the owned services, select the recorded known-good revision in the deployment checkout, restore its service definitions and start again. Preserve runtime SQLite files, keys, model files and Ratmir's original checkouts.

## Jury contract and remaining gates

`POST /api/v1/analyze` accepts multipart field `file` with a MOV and `X-API-Key`; acceptance returns 202 plus a job ID. `GET /api/v1/jobs/{job_id}` uses the same key. Completed results include the two verdict values, event types/times/descriptions and 1–3 base64 JPEG frames per event. Errors use `failed`; overload uses 429 with a retry hint. One job is processed at a time.

The synthetic workflow and real positive/negative train uploads were checked locally on this host. Only aggregate diagnostics were returned to the coding assistant. Humans still need to inspect private evidence and rule applicability themselves.

Model inference is operational, but detection quality is not accepted for competition: Ratmir's Bonsai validation report has TP=1, TN=3, FP=1, FN=2 across seven unique videos, F1=0.400. Passing deployment and schema checks does not establish semantic event accuracy. His final weak CPU holdout was already used and must not be reused for tuning. See `docs/ml-handoff.md` and `reports/ml-quality.json`.

Public HTTPS publication remains a separate gate. The earlier attempt to start Cloudflare Tunnel was rejected by automatic tool approval with “blocked by policy,” without a specific reason. This deployment does not move that denied publication action to SSH or another wrapper. No public URL or successful external jury upload is claimed. A permitted publication workflow and an independent-network end-to-end test are still required before submission.
