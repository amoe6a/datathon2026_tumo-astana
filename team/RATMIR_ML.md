# Ratmir's Codex prompt for detection and evaluation

Open the shared `forte-review` repository in Codex and paste the task below. You need the whole code folder/ZIP, not just this prompt. Keep the shared root `AGENTS.md`; use this file by explicit reference.

---

You are helping Ratmir own ML for a three-person ForteBank Datathon team. The other teammates own backend/deployment and frontend/presentation. We each have 12 productive hours. Build a defensible detector behind the existing interface; do not rebuild the app or spend the MVP window training a large model.

Read `AGENTS.md`, `README.md`, `HACKATHON_PLAN.md`, `ml/`, `rules/catalog.json`, `data.example.json`, `scripts/evaluate.py` and current tests. The software already runs a synthetic end-to-end demo. It includes a trainable CPU reference classifier and an unverified-on-GPU Qwen3-VL adapter. There is no real-data accuracy result yet. Inspect code before replacing it.

## Constraints

All runtime models must execute locally inside our service, offline after weight download. No cloud LLM/vision/embedding service. No bank footage, extracted frames, descriptions of private frames or detailed per-video analysis may be exposed to Codex or another cloud assistant. Execute local scripts and return only aggregate counts/timings/errors. Ask me to inspect videos myself in a local player and enter labels into local files; do not use image tools on bank frames. Code and synthetic geometric scenes can be used for debugging.

No face identification or audio. Only visible actions may support a possible event. Never infer theft, accounting errors, exact money amounts, identity correctness or invisible system state. A model error or missing model must not turn into `no_violation`. Keep two allowed verdicts; visibility uncertainty is metadata, not a third verdict.

## Hardware and setup

Team hardware: RTX 4060 laptop, 8 GB VRAM, 32 GB RAM, i9-14900HX. We have about $30–35 of Brev GPU credit, but local inference is the first choice. Do not assume 8 GB guarantees a given batch/context size. Measure GPU memory and cold latency.

Create/reuse Python 3.11 `.venv`, install core requirements, run the synthetic tests. On the RTX laptop only:

```powershell
.venv\Scripts\python.exe -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
.venv\Scripts\python.exe -m pip install -r requirements-gpu.txt
.venv\Scripts\python.exe -m scripts.doctor
.venv\Scripts\python.exe -m scripts.download_model
```

Verify CUDA and imports before full data processing. Confirm the driver supports the chosen wheel. Read the publisher's Qwen3-VL-2B-Instruct card/license; retain license and pinned revision. The supplied adapter uses FP16, bounded resolution, overlapping windows, deterministic generation and local files only. It has not been run on our target GPU yet. Cap setup troubleshooting at 45 focused minutes before returning to the baseline.

## Ownership and contract

Own `ml/`, `rules/catalog.json`, `scripts/evaluate.py`, `data.example.json`, model setup and `docs/ml-handoff.md`. Amir audits normative sources and gives you source notes; you integrate reviewed mapping. Backend teammate owns routes, worker supervision and deployment. Do not change those routes or `web/`. Propose shared dependency changes before committing them. Use branch `ml` in a separate clone; no force-push.

Keep `analyze(path, model, engine, rule_path, progress)` returning the current result schema. Events require actual source timestamps, clear visible observations and 1–3 JPEG frames. The model may return only allowed rule IDs. Rules come from the catalog, never generated URLs. An isolated frame cannot establish an omitted action; preserve full temporal coverage and caveats.

## My 12-hour work queue

1. **0–0.5 h:** Run CPU demo; inspect GPU and imports. Agree the inference contract and first acceptance videos with the team.
2. **0.5–2 h:** Start the model download; label my third of the dataset locally while teammates label theirs. Build the inventory and group related sessions before splitting. Do not assume the case's categories are already known.
3. **2–5 h:** Fit the CPU baseline, run Qwen on one short positive and one normal clip, add categories actually observed in training with valid rules, and compare behavior. Integrate the better feasible candidate before optimizing it. Return counts and latency only.
4. **5–6 h:** Freeze a candidate config and provide model path, class list, revision/hash and inference timings for API integration. Verify multiple events survive the output adapter.
5. **6–8 h:** Evaluate on validation groups, review mistakes locally with teammates, measure cold longest-clip latency and peak VRAM. Decide whether the model is useful beyond memorizing background.
6. **8–10 h:** Make one change targeted to the largest measured error: better labels, threshold, sampling or prompt. Compare on the same validation set. Preserve the previous candidate. No uncontrolled hyperparameter sweep or large fine-tune.
7. **10–11 h:** Freeze weights/config, evaluate the untouched test once, document failure modes and unsupported classes. Report counts and denominators, not unsupported confidence claims.
8. **11–12 h:** Verify the frozen model runs in the submitted release; rehearse 30–45 seconds on model selection, evidence and limitations. Support Q&A.

## Dataset workflow

Ask for the authorized local dataset folder if not already configured. At least 40 staged overhead MOVs are described by the brief; organizer labels are binary plus an event time, without event types. Human reviewers must define types and intervals. Each teammate does roughly one third; ask them to take extra annotation if necessary rather than losing all my model time.

Copy `data.example.json` to an ignored `data/manifest.json`. Group clips by session/actors/camera; keep near-duplicates together. Aim at 60/20/20 train/validation/test when counts permit. Never random-split frames. Use explicit normal intervals; do not label all frames outside a positive timestamp as normal. Preserve ambiguous intervals as unlabeled. If events overlap, the current reference classifier cannot train multi-label segments; document this and use a suitable representation rather than silently choosing one class.

```powershell
.venv\Scripts\python.exe -m ml.train data/manifest.json --output models/reference.npz --threshold 0.8
.venv\Scripts\python.exe -m scripts.evaluate data/manifest.json --split val --output runtime/reference-val.json
.venv\Scripts\python.exe -m scripts.evaluate data/manifest.json --engine qwen --model models/qwen3-vl-2b --split val --output runtime/qwen-val.json
```

Select a threshold on validation; then freeze it. The included evaluation reports clip confusion counts, failed-video rate, event/type matching under an internal 3-second tolerance and timing. This tolerance is not from the organizers. Confirm exact frame/time manually. Full API cold latency includes model load/upload; have the backend teammate measure it. Small samples require reporting counts and limitations.

## Rule and model choices

Starting hypotheses: cashier leaves during ongoing service (NBK 231 paragraph 22); previously received cash left exposed during the next transaction (paragraph 20). They are not known dataset labels. The catalog also contains disabled counting/bag-inspection omission candidates; enable only if the full action and exceptions can be handled. Payout counting exceptions in paragraph 32 matter. Do not use vague “suspicious behavior” as a legally sourced class.

Reference mode is a lightweight appearance/motion classifier. Qwen is the semantic candidate. Neither should be trusted without held-out evidence. The final video may use a different angle and unknown event types. Test actual angle variation if available; crops/rotations are limited stress tests. Unknown classes remain a limitation rather than invented detections.

If the 2B model is insufficient, propose one short Brev experiment with a clear hypothesis, validation set, cost cap and stop condition. Do not upload event data to Brev until organizers confirm team-controlled VM use is allowed. No training from scratch; no hosted inference APIs. Do not choose Qwen2.5-VL-3B merely because it is small: its license differs from the selected Qwen3 model and must be checked independently.

## Definition of done

Deliver reproducible local training/inference commands, tested weights/config, held-out counts, latency/VRAM measurements, supported classes with rules, known failures and a fallback. `docs/ml-handoff.md` must distinguish measured facts from assumptions and synthetic from real results, include hashes/versions and the next unresolved action, and contain no event media, private frame descriptions or API secrets. Do not declare success because the model loads or gives plausible prose.

Start by running the existing tests and checking the GPU environment. Continue independent code/setup work while waiting for data; do not invent labels or results.
