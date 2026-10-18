# Deadline-bounded Qwen upgrade check — 18 October 2026

**Decision: retain the published Qwen3-VL-2B scene-LoRA release.** The larger download is not a usable drop-in upgrade, and the completed updated-annotation 2B experiment is worse on comparable validation. No candidate was promoted. Inference code, selected weights, adapter, rules and `configs/qwen-release.json` remain unchanged.

Machine-readable aggregate evidence: [`reports/qwen-upgrade-20261018.json`](../reports/qwen-upgrade-20261018.json). Historical reports have not been overwritten.

## Time and initial deployment

The first clock check was **11:27:16 Asia/Qyzylorda (UTC+05)**. Submission is recorded as noon; experiments were frozen by **11:40**, protecting the final 20 minutes. Availability remains required through **17:00**.

The checkout was clean at `99ba7062f77855f35b7262a6cf13c0d8bae0a994`. Jury, ngrok and sleep-inhibitor user services were active; Bonsai remained disabled. Despite an active API process, readiness was initially **503**. There were 20 completed jobs and no active jobs.

The unchanged baseline passed a bounded offline synthetic probe in **5.2 seconds with zero network attempts**. Readiness cached a failed startup probe and did not retry automatically. One idle-service restart at **11:35:51** recovered it; local and public readiness were verified **200 / ready** at **11:36:26**. The original triggering exception is unknown: the probe suppressed its diagnostic details. Earlier resource contention is only a possibility, not a confirmed root cause. No driver, dependency, model or runtime-environment upgrade was performed.

The service unit, environment, key, selected config, catalog and database hashes were preserved across recovery. All 20 original completed jobs remained readable. The subsequent successful public test adds one completed job.

## Actual larger checkpoint

Discovered download:

`/home/angel/Загрузки/Qwen3-VL-8B-Instruct-UD-Q5_K_XL.gguf`

- **Qwen3-VL-8B-Instruct**, architecture `qwen3vl`, size label **8B**, not the older 4B model.
- GGUF v3, 5,884,767,072 bytes, 399 tensors. Filename identifies Unsloth **UD-Q5_K_XL**; tensor storage is mixed, not uniformly five-bit.
- SHA-256: `d138d047fe5736cdc678490d6ba16d2e42f690a6d648b5c8fbdac0e17789fac6`.
- Metadata attributes quantization to Unsloth and identifies `Qwen/Qwen3-VL-8B-Instruct` with `apache-2.0`. **The exact publisher revision and artifact provenance could not be established locally.** These are metadata claims, not an independently verified license/provenance audit.
- Tokenizer data and chat template are present; image/video token markers are present. This identifies a vision-language family but does not prove a complete working vision installation.
- The bounded parser checked known tensor layouts, alignment, overlap and payload bounds; there are no unreferenced payload bytes. It did not validate tensor values or model behavior.
- **No matching vision projector was found** in the bounded download/model searches. The existing Bonsai projector is not interchangeable.

The current pipeline uses Transformers/PEFT local checkpoints. This GGUF needs a compatible local multimodal GGUF runtime and its matching vision projector, followed by chat-template, constrained-output, sampling, timestamp and rule-filter integration tests. The existing local `llama-server`/`llama-mtmd-cli` binaries have **not** been validated with this pair. Do not attach the 2B LoRA, feed the GGUF directly into the PEFT trainer, or treat text-only generation as a visual pilot.

**No 8B inference, training, validation, CUDA-memory or latency measurement ran.** Building and validating a separate runtime would not fit the remaining pre-freeze budget. Its visual pilot is explicitly blocked rather than recorded as successful.

Reusable metadata preflight, requiring a new output path:

```bash
.venv/bin/python -m scripts.inspect_qwen_candidate \
  --model /home/angel/Загрузки/Qwen3-VL-8B-Instruct-UD-Q5_K_XL.gguf \
  --output runtime/new-8b-preflight.json
```

An optional `--projector` inspects its structure too; it does not establish model/projector compatibility.

## Updated workbook and preserved splits

Selected workbook: `/home/angel/Рабочий стол/datathon/open/open_labels.xlsx`, identical by hash to the private existing upgrade snapshot.

- SHA-256: `401ca60d0ce3bbabca3bd0028d610716f9f103fd3ab23b03afe51fe11775fe38`.
- 42 rows, 40 unique video hashes; **26 train / 7 validation / 7 exposed test**, unchanged by content hash.
- Descriptions changed for **16 videos: 8 train, 5 validation, 3 test**.
- **No binary-label or timestamp-anchor changes**, missing videos, contradictory binary labels, or invalid/out-of-bounds timestamps were found. All 40 video durations were checked locally.
- Two extra duplicate rows and two duplicate groups with description variants remain. Their semantic consistency requires local human review.
- Free-text changes have not been semantically classified into actor/action/rule-reference corrections. No actor/action accuracy or reviewed rule-label accuracy is claimed.
- No new recordings or fresh independent holdout were found. The old seven-video test is exposed; **it was not rerun in this session** and was not used for selection.

The original manifest is intact at `runtime/qwen-training-20261017/dataset.json`, SHA-256 `8b978786b7ddc15d47a3bf1fb5051c0d48d7c9bae78e1fcaf1a11eeec24f1d1f`. The original workbook bytes matching its historical recorded hash were not recovered; existing older workbooks do not match that hash. Do not present a different workbook as the original.

Two distinct private versions are preserved:

1. Existing **training-compatible schema v1**: `runtime/qwen-upgrade-20261018T060651Z/dataset.json`, hash `07b991d373274f9a8d0a39be503f7f075d528c2319ca9f311632bf2d7c7be644`.
2. New **audit-only schema v2**: `runtime/qwen-annotation-audit-20261018T1128-v2/manifest.v2.json`, with a separate aggregate `summary.json`. It preserves source-row relationships, explicit unknown supervision and review flags. **Do not pass it directly to the trainer.**

The new `scripts.audit_qwen_annotations` CLI requires an explicit workbook and expected hash, rejects overwriting output, and stores private manifests only under ignored `runtime/`. `--help` documents its arguments. Missing descriptions never become negative labels. Annotation text and generated observations remain local.

## Comparable validation and existing training

A completed four-epoch updated-annotation 2B scene-LoRA run was already present when inspected. Its status, four checkpoints, training-manifest hash, evaluation source hashes and complete seven-video report were verified. It trained on **26 training videos only**, took **110.548 seconds**, and peaked at **5,618,987,520 allocated CUDA bytes**. This session did not rerun training. Loss decreased from 2.2600 to 1.5011, but detection worsened.

Baseline predictions were reused from the hash-verified `val-release-parity/report.json`: all seven video hashes and revised binary labels match. The release config, inference source, base weights, selected adapter and every frozen rule file match the frozen release hashes. Inference knobs match the updated-2B evaluation after resolving paths; the adapter differs. Thus this comparison does not mix corrected candidate labels with original baseline labels.

| Same seven validation videos | TP | TN | FP | FN | Failures | Precision | Recall | Specificity | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Serving 2B adapter, saved predictions re-scored | 2 | 3 | 1 | 1 | 0 | 0.667 | 0.667 | 0.750 | **0.667** |
| Existing updated-annotation 2B adapter | 2 | 0 | 4 | 1 | 0 | 0.333 | 0.667 | 0.000 | **0.444** |
| Downloaded 8B GGUF | — | — | — | — | — | — | — | — | **Not evaluated** |

| Operational validation statistic | Baseline | Updated 2B |
|---|---:|---:|
| Model load seconds | 3.946 | 4.475 |
| Warm median / maximum seconds | 11.275 / 33.620 | 34.643 / 52.479 |
| Peak CUDA allocated bytes | 5,620,063,232 | 5,700,824,064 |
| Median nearest-anchor error, seconds | 26.250 | 27.2335 |
| Videos contributing to anchor statistic | 2 | 2 |

The anchor statistic is **not semantic localization accuracy**; it uses the nearest annotation anchor among events on detected positive videos only. Validation was repeatedly inspected, so it is not an unbiased performance estimate. The historical exposed-test result remains in the original release report; there is no new independent score. No improvement can be attributed to model size because the 8B never ran.

## Actual final checks

- **177 Python tests passed**, Python 3.11 project environment, 24.48 seconds, one deprecation warning.
- **177 Python tests passed**, unchanged Python 3.14 release environment, 25.00 seconds, two deprecation warnings.
- `scripts.doctor`, frontend JavaScript syntax, and **12 frontend tests passed**.
- Public HTTPS upload of the **142.533-second / 28,718,927-byte longest training MOV** completed in **97.222 seconds**, including upload, cold worker model loading and polling. Upload acceptance took **27.526 seconds**.
- Returned `done`, two events, five JPEGs: all decoded into pixels; timestamp bounds, required schema and official-source metadata passed. Missing/wrong keys were rejected on upload and polling routes.
- The real-video test is transport/schema evidence on training data, **not an accuracy estimate**. HTTPS was tested from this host, not an independent physical network.
- Restart recovery, unchanged key and completed-job persistence were actually checked. The tunnel and public URL were not changed. Bonsai remains disabled and keyless testing remains off.

## Serving and rollback

Public address: **https://fracture-freestyle-unvalued.ngrok-free.dev**.

Private key file: `/home/angel/forte-review-service/runtime/forte-jury/access-key.txt`. Do not print, commit, or send the key to the coding assistant.

Selected detector: `configs/qwen-release.json`, `models/forte-qwen-release`, inference source `51a54a052a2ea43b455f5ef93e7a5e4ade8e6ef7`; published source/packaging baseline `99ba7062f77855f35b7262a6cf13c0d8bae0a994`. The later audit/report commit does not change inference.

Private pre-recovery backup: `runtime/baseline-recovery-20261018T063430Z/backup/`, containing `service.unit`, `service.env`, `access-key.txt`, `qwen-release.json`, `rule-catalog.json` and a database snapshot. No candidate switch occurred, so the service already is the rollback target.

If recovery is needed, first check `/health` and ensure no upload/job is active; restarting interrupts active work. For an unchanged deployment, restart only the owned jury service, then check local/public readiness and authenticated job access:

```bash
systemctl --user restart forte-jury.service
curl --fail http://127.0.0.1:8000/ready
curl --fail https://fracture-freestyle-unvalued.ngrok-free.dev/ready
```

If someone subsequently changes the unit/config/environment, privately restore `service.unit` to `/home/angel/.config/systemd/user/forte-jury.service`, `service.env` to `runtime/forte-jury/service.env`, `qwen-release.json` to `configs/qwen-release.json`, and `rule-catalog.json` to `runtime/forte-jury/rule-catalog.json` from the backup, then run `systemctl --user daemon-reload` and restart only when idle. Preserve 0600 permissions on private files. **Do not replace the current database with the snapshot**: doing so would discard newer completed jobs. Do not regenerate the key. Keep the backup private; it contains credentials and private results.

The previously documented older full-clip fallback remains available in [Linux deployment](linux-deployment.md). Do not start Bonsai or substitute demo weights.

## Remaining handoff

Keep both laptops powered, awake and connected until **17:00 UTC+05**. The team must confirm organizer receipt of the unchanged URL/private key and remaining ngrok quota. No submission or quota check is claimed. Ratmir should inspect duplicate description variants and disputed validation examples locally; only example numbers/error categories should enter chat. Additional 8B work requires permission outside the protected judging deployment window, a verified matching projector/runtime and complete comparable validation before promotion.
