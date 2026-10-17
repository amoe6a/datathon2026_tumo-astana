# Qwen release evidence — 18 October 2026

The selected detector is **Qwen3-VL-2B-Instruct with the fourth-epoch scene LoRA, BF16 inference and denser full-clip verification**. Configuration: `configs/qwen-release.json`. Inference source was frozen at `51a54a052a2ea43b455f5ef93e7a5e4ade8e6ef7`; later report and packaging commits do not change the detector.

The public UI/API is **https://fracture-freestyle-unvalued.ngrok-free.dev**. [Judge access and recovery](judges-handoff.md), [Linux deployment](linux-deployment.md), [short pitch](pitch.md).

## Measured detection

| Split | TP | TN | FP | FN | Failures | Precision | Recall | F1 | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Validation, 7 videos, release-environment reproduction | 2 | 3 | 1 | 1 | 0 | 0.667 | 0.667 | **0.667** | 5/7 = 71.4% |
| Held-out test, 7 videos, once after freeze | 2 | 0 | 3 | 2 | 0 | 0.400 | 0.500 | **0.444** | 2/7 = 28.6% |

**Generalization is weak:** all three negative test clips produced false positives, and two of four positive clips were missed. The software is a working review prototype; these results do not support reliable automatic compliance decisions. We did not tune the model against these test answers.

Validation warm inference took a median 11.275 seconds and maximum 33.620 seconds. Held-out warm inference took a median 24.301 seconds and maximum 42.240 seconds. These figures exclude HTTP upload, model loading and API polling; see the public end-to-end checks below. Conventional median nearest-anchor error was 26.25 seconds on two detected validation positives and 16.5 seconds on two detected test positives. These are different subsets, not a measured localization improvement.

Machine-readable evidence: [all aggregate experiments](../reports/qwen-experiments.json) and [frozen release / operational evidence](../reports/qwen-release.json). Detailed predictions and footage remain private on the RTX host.

The target is the organizer's **per-video binary label**. F1 is not accuracy, actor recognition accuracy or event classification accuracy. Multiple configurations were selected on only seven validation videos, so validation optimism is substantial. Qwen's final test ran once after freezing code, configuration, rules and adapter. An earlier CPU comparator had already evaluated this split; it is not a pristine benchmark for the entire project. Exact duplicate videos are grouped, but camera/session independence has not been established.

The selected model produces no events on four validation videos: three are correctly negative and one is a miss. Empty findings therefore do not establish that a recording is compliant. This is separate from an inference failure, which the API reports as `failed`, never as `no_violation`.

## Why this model was selected

All rows below are completed seven-video validation runs with zero processing failures. The complete aggregate export includes unsuccessful and partial experiments as well as the selected candidate.

| Candidate | TP / TN / FP / FN | F1 | Maximum warm seconds | Decision |
|---|---:|---:|---:|---|
| Matched BF16 2B base, refined pipeline, no adapter | 0 / 3 / 1 / 3 | 0.000 | 34.825 | Missed every positive |
| 2B scene LoRA, original full-clip pipeline, reproduction | 2 / 3 / 1 / 1 | 0.667 | 18.532 | Working fallback |
| 2B scene LoRA, dense full-clip verification | 2 / 3 / 1 / 1 | 0.667 | 33.543 | Selected; same binary result, modest timing diagnostic improvement |
| 2B scene LoRA, short windows | 2 / 0 / 4 / 1 | 0.444 | 76.818 | Too many false positives |
| Short windows plus full-clip re-verification | 2 / 1 / 3 / 1 | 0.500 | 100.520 | Inferior to selected pipeline |
| 4B scene LoRA, BF16 frozen components, epoch 4 | 0 / 4 / 0 / 3 | 0.000 | 17.503 | Missed every positive |
| 2B timeline supervision, epochs 2 and 4 | 0 / 4 / 0 / 3 | 0.000 | See aggregate export | Missed every positive |

A separate direct binary-triage experiment reached F1 0.333, with TP=1/TN=2/FP=2/FN=2. It is a diagnostic classifier, not a complete rule-grounded detector, and is not deployed. The original 4B training pilot ran out of memory; keeping frozen components in BF16 made training fit, but did not improve detection. Larger models and lower training loss were not treated as evidence of better results.

An earlier base-model comparison mixed 4-bit base inference with BF16 adapted inference. It cannot isolate the effect of training. The completed matched BF16 control above addresses that confound for the selected pipeline; it remains a small-sample, single-seed comparison.

## Training and frozen assets

- Dataset: 42 workbook rows, 40 unique video hashes; 26 train, 7 validation, 7 test. Training has 14 positive and 12 negative videos.
- Human free-text annotations supervise scene descriptions of actors, actions and violations. They are not exhaustive event labels, bounding boxes or independently validated rule labels. No generated pseudo-labels or test examples were used for this adapter.
- Four epochs of QLoRA, NF4 double quantization, rank 8, alpha 16, dropout 0.05, language attention query/value projections; frozen vision encoder and base weights. Learning rate 0.0001, gradient accumulation 4, seed 20261017; 1,605,632 trainable parameters.
- Training loss fell from 2.5783 to 1.7538; training took 111.017 seconds and allocated at most 5,590,366,720 CUDA bytes. These training figures are not detection scores.
- Publisher: `Qwen/Qwen3-VL-2B-Instruct`, revision `78448d793a7eb2f7a987a1da76d464384aa1becd`, Apache 2.0. Weight SHA-256: `7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0`.
- Selected adapter SHA-256: `50bd5d6a0541958c67b3106aecc0deb8d20f7517906e3b207926bce46938b03f`. It stays at `models/forte-qwen-release` on the RTX host, outside Git and the team kit.
- Inference decodes at 2 FPS, selects 32 images across up to 180 seconds, retrieves four local rule cards, and verifies proposed events on 48 more densely sampled images. The adapter is used for observations and decisions. This can miss brief actions and cannot prove an omitted action merely because it was not sampled.
- The supplied archive contains 218 source chunks and four official sources. Runtime retrieval searches its 29 curated rule cards: 15 conditional and 14 review-only. Review-only cards cannot independently create automatic violations. Rule conditions and exceptions remain explicit; model self-reports of visible conditions can still be wrong.

## Operational verification

- **115 Python/API tests passed on Linux**, 26.737 seconds, zero failures/errors/skips, in the independent `.venv-release` environment. Earlier Windows verification passed 112 full-suite tests; subsequent targeted RAG/evaluation checks passed 27 tests. **12 frontend checks passed**.
- Actual local model load, Russian constrained JSON and complete synthetic analysis passed with socket connections blocked: **zero network attempts**, 40.584 seconds. This covers inference without internet; the public tunnel itself naturally requires internet.
- The release environment reproduced every validation video's binary verdict before the test was opened. All seven test videos processed successfully. Inference source/configuration hashes remain unchanged after freeze.
- The active stack is Python 3.14.8, Torch 2.14.0+cu130, Transformers 5.17.0, PEFT 0.21.2 and the pinned LM Format Enforcer Git revision `817f944fcc8851917c40300a47f62d1defc3ffc3`. The release environment has its own copied packages; base weights and adapter are inside the dedicated project, not linked to the original ML checkout.

| Final API check | Input | Route | Upload-to-result seconds | Events / decoded JPEGs |
|---|---|---|---:|---:|
| Short positive training clip | 14.700 s MOV, 3,044,364 bytes | Loopback API | 24.057 | 1 / 3 |
| Short negative training clip | 15.867 s MOV | Loopback API | 21.043 | 0 / 0 |
| Longest training clip | 142.533 s MOV, 28,718,927 bytes | Public HTTPS | **101.800** | 2 / 5 |

These functional examples are from training data and establish transport/schema behavior, not an independent accuracy estimate. The public longest-clip run included 36.386 seconds accepting the upload, model loading, inference, evidence generation and polling; it met the 300-second limit. Missing/wrong keys were rejected. The verifier decoded every evidence JPEG into pixels and checked finite, in-range timestamps and official source metadata.

A **29,000,000-byte MOV upload from the Windows laptop through public HTTPS completed in 112.688 seconds**, including 95.016 seconds accepting the upload. This was a ten-second H.264 synthetic fixture padded to 29 MB for transport testing; it is not a long-content or accuracy test. It returned a valid `done / no_violation` response. The separate real 142.533-second test above covers long-content processing.

The public browser completed the synthetic H.264 upload and polling workflow, displayed the result in approximately 17 seconds, and showed the limitation that no detected events do not establish compliance. [Public UI proof](frontend-preview/final-public-synthetic-browser.png) uses synthetic content only. The separate CPU demo verified evidence enlargement; its MPEG-4 preview used the documented fallback, while the public H.264 preview decoded successfully.

API restart preserved the key and completed jobs; tunnel restart preserved the public address. Keyless testing is disabled, Bonsai is disabled, and the sleep inhibitor is active. The public browser loaded all 29 authenticated rule cards and correctly decoded a synthetic H.264 MOV preview. Synthetic CPU-demo evidence enlargement and source inspection were also checked; the demo is not a model-quality benchmark.

Training and private review outputs remain on Ratmir's laptop. Source-only distribution includes public code, rules, configuration, documentation and aggregate results. It excludes footage, private annotations, generated observations, evidence, adapter/base weights, virtual environments and credentials. The selected validation review is available locally at `/home/angel/Рабочий стол/Forte-Qwen-CURRENT-REVIEW.html`.

## Limits and next work

Actor identification here means visible roles such as cashier or client, not a person's identity. Role accuracy, per-rule accuracy, exception handling and event-matched localization have not been independently scored. The nearest-anchor statistic finds the closest organizer timestamp among predicted events; it can look good for the wrong event, uses only detected positive videos, and must not be presented as localization accuracy. Historical summaries used an upper-middle value for even samples; the final export corrects this to the conventional median without changing original reports.

Unseen rule types and a new camera angle are architectural goals, not demonstrated generalization. False positives and missed violations remain. The next defensible improvement requires reconciled event-level annotations, examples of prerequisites/exceptions, additional cameras and a new independent evaluation set. Do not tune against the final seven-video test or judge uploads.

The endpoint has recovery services and sleep inhibition, but power, the Windows hotspot and provider quota remain external dependencies. Keep both laptops plugged in and connected through **18 October 2026, 17:00 UTC+05**. Remaining ngrok quota and a full upload from an independently connected external network are unverified. Submit the URL and private key through the organizer's official channel; this project does not automatically submit them.
