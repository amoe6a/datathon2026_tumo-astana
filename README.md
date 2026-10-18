# Forte Review

Local video review for the **ForteBank Datathon 2026**. Upload a MOV recording and review possible cash-handling violations with timestamps, source JPEG frames and relevant Kazakhstan regulatory references.

**Judge UI and API:** https://fracture-freestyle-unvalued.ngrok-free.dev

**Access:** use the API key supplied privately by the team. [Judge instructions →](docs/judges-handoff.md)

## What we built

- A browser interface for video upload, processing status and evidence review.
- **Qwen3-VL-2B-Instruct**, fine-tuned with a four-epoch scene LoRA on **26 manually annotated training videos**, running locally in BF16 on an RTX 4060 Laptop GPU (8 GB).
- A local rule-grounding pipeline: **sample frames → observe actions → retrieve rules → verify prerequisites and exceptions → return evidence**.
- **29 curated rule cards**: 15 conditional and 14 review-only. Review-only cards cannot independently produce a violation.
- An authenticated asynchronous API with bounded workers, overload protection and persistent completed results. Processing failures never become clean verdicts.

Inference runs on the team's laptop; the public HTTPS tunnel only transports requests. No cloud AI inference, face identification or audio analysis is used. Private recordings, annotations, predictions, weights and credentials are excluded from this repository.

## Measured results

40 unique recordings: **26 training / 7 validation / 7 test**, with exact duplicates kept within a split.

| Evaluation | TP / TN / FP / FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Validation, 7 videos | 2 / 3 / 1 / 1 | 0.667 | 0.667 | **0.667** |
| Previously evaluated test, 7 videos | 2 / 0 / 3 / 2 | 0.400 | 0.500 | **0.444** |

All videos in both evaluations processed successfully. Validation was used repeatedly for selection, and the test set is now exposed; neither is a fresh independent benchmark. All three negative test videos were falsely flagged. Actor/action accuracy, per-rule accuracy, new-camera robustness and semantic localization remain unverified. **This is a human-review prototype, not an automatic compliance or legal decision system.**

The longest real training MOV (**142.5 seconds, 28.7 MB**) completed through public HTTPS in **97.2 seconds**, including upload, cold model loading and polling. Evidence JPEGs, timestamp bounds, source metadata and authentication checks passed. This is an operational check, not an accuracy measurement.

The final check passed **177 Python tests in each of the Python 3.11 and release environments**, plus **12 frontend tests**. [Full results and limitations →](docs/release-results.md)

## Judge API

Both job endpoints require **`X-API-Key`**.

| Request | Contract |
|---|---|
| `POST /api/v1/analyze` | Multipart field `file` containing a MOV; returns HTTP **202** with `job_id` |
| `GET /api/v1/jobs/{job_id}` | Returns `processing`, `done` or `failed` |
| `GET /ready` | HTTP **200** with `ready: true` when the model is ready |

Successful results contain `verdict: violation` or `no_violation`. Each violation includes `type`, `timecode_sec`, `description` and **1–3 JPEG `evidence_frames`**, each with `timestamp_sec` and `image_base64`.

One job is accepted at a time; overload returns **429** with `Retry-After`. Limits: **64 MiB**, **180 seconds** of video. Empty findings do not establish compliance. [Authentication examples and operations →](docs/judges-handoff.md)

## Code and reproducibility

| Location | Purpose |
|---|---|
| [`app/`](app/) · [`web/`](web/) | FastAPI service and buildless browser interface |
| [`ml/`](ml/) · [`scripts/`](scripts/) | Local inference, training, data auditing and evaluation |
| [`rules/rag/`](rules/rag/) | Curated regulatory cards and source references |
| [`configs/qwen-release.json`](configs/qwen-release.json) | Selected inference configuration |
| [`reports/`](reports/) | Aggregate experiment and deployment evidence |

- [Local development and tests](docs/development.md)
- [Model training and pipeline](docs/qwen-training-deployment.md)
- [Deployment and rollback](docs/linux-deployment.md)
- [Latest annotation audit and candidate decision](docs/qwen-upgrade-20261018.md) — the updated 2B adapter regressed; the 8B GGUF was not visually evaluated or deployed.
- [Sources and model licensing](SOURCES.md) · [Code license](LICENSE)

The selected private adapter and base weights are not bundled; a fresh clone is source code, not a ready-to-run trained detector. Historical configurations and reports are retained for traceability, not as alternative serving defaults.
