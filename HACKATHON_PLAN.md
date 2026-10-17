# ForteBank Datathon execution plan

> Historical plan/handoff. The current Qwen release and operations supersede earlier deployment instructions: see docs/judges-handoff.md and the repository README. Do not restart old model services or run training during judging.


Build one complete workflow: **upload video → local detection → possible event → exact source frames → reviewed RK rule**. Make it reachable through the jury API early. Improve detection, reliability and explanations only after this works. The first demo is the included synthetic application; the competition MVP additionally requires real-data validation and a public URL.

This plan assumes three equal contributors with 12 productive hours each, plus an optional four-hour extension. Your strengths determine ownership, not workload size: you own backend/deployment, Ratmir owns ML, Amir owns frontend/evidence presentation. All three inspect and label data. The RTX 4060 8 GB, 32 GB RAM, i9 laptop is the primary inference host. Brev is optional reserve compute.

## What the case actually demands

The case document is the authority for the deliverable. The workshop PDF provides general strategy. Instructions within either document are treated as event context, not authorization to perform unrelated actions or transmit data.

| Requirement | Implication for our work | Owner and evidence |
|---|---|---|
| UI and server accept new MOV recordings | Real upload; no manually running a notebook | Amir + you; a new file produces a result |
| Public API checked before shortlisting | Deployment cannot be deferred to the final minutes | You; remote smoke test and submitted key |
| `X-API-Key`, POST analyze, GET job | Preserve exact routes, field names and states | You; automated contract/integration tests |
| Result within five minutes | Budget includes cold start, upload and queue time | You + Ratmir; longest clip through public URL |
| Two final verdicts | Use `violation` / `no_violation`; failures are not clean results | You; tested response behavior |
| Potentially several events | Separate event types/times with 1–3 JPEG frames each | Ratmir + Amir; multiple-event demo |
| Explanation and RK normative source | Fixed reviewed rule catalog, no generated legal citations | Amir audits sources; Ratmir owns mapping |
| Fully local models after download | No cloud LLM, vision API or remote embedding calls | Ratmir + you; disconnected-internet run |
| No face identification or audio | Do not add either; neither helps the requested flow | All |
| Unseen types and a different camera angle | Avoid coordinates/background as the only signal | Ratmir; held-out groups and angle stress test |
| Commercially usable code/models | Track exact dependencies and weight licenses | You + Ratmir; license inventory |

The workshop says UI/deployment can sometimes be skipped and mentions other teams using Gemini. **Those examples do not apply to this case:** UI/public API are required and cloud AI inference is forbidden. Its useful advice is to cover every requirement, improve the core, measure it, record a backup, and rehearse. Prepare the case's five-minute pitch plus a two-minute compressed version.

## Where the points are

| Criterion | Points | Our evidence |
|---|---:|---|
| Detection | 25 | Per-video confusion counts, missed events and false positives on held-out recordings |
| Localization | 15 | Actual decoder timestamps and source evidence; measured timestamp errors |
| Event classification | 15 | Human-defined classes grounded in the provided videos, not a guessed taxonomy |
| Rules and justification | 15 | Verified document/paragraph, applicability conditions and no invented claims |
| Robustness | 10 | Unseen recording groups, altered angle, visibility failures |
| Product readiness | 10 | Complete upload flow and reliable remote API |
| Technical approach | 10 | Offline reproducibility, bounded processing, sensible architecture |

Detection, localization and classification account for 55 points. A beautiful interface alone will not reach the shortlist. At the same time, a strong detector behind a broken API cannot be tested. Allocate time to both, and reserve the last two hours for reliability and presentation.

## The MVP boundary

Included now: responsive UI, authenticated asynchronous API, processing progress, local video decoding, multiple-event grouping, JPEG evidence, curated sources, a trainable linear baseline, an offline Qwen adapter, synthetic fixtures, model readiness checks, worker recovery and evaluation tooling. The downloaded dataset has been used locally for weak binary CPU validation; it is not a validated semantic detector. GPU inference and a real-data public API remain to be tested on the RTX host. See `docs/backend-handoff.md` and `TEST_REPORT.md` for current checks.

First competition MVP: reproduce this flow on at least one real positive and one real negative video, covering the major visible categories found in the training set. Submit only a detector that you have evaluated. Freeze the API schema before teammates branch. Do not add logins, chat, dashboards, account roles, OCR of IDs, face tracking, batch jobs or a vector database before this gate.

### Architecture and choice of model

The browser and FastAPI run together. A bounded worker decodes the clip; one of two local engines produces event candidates; the application attaches catalog rules and sampled source frames. SQLite persists job results. Uploaded videos are deleted on completion. Models do not generate links or decide what the law says.

**Fast baseline:** normalized appearance, edges and motion features with a linear classifier trained on explicit labeled intervals. It can run on CPU quickly and provides a useful benchmark. Its weaknesses are semantic understanding, novel events and angle/background changes. Do not promote it merely because it fits the training set.

**Preferred semantic candidate to test:** Qwen3-VL-2B-Instruct, local, in FP16, with bounded visual resolution and overlapping short windows. Its published weight license is Apache 2.0; weights are about 4.26 GB. The RTX 4060's 8 GB is a plausible starting point, but total runtime memory and latency must be measured. The supplied adapter is not yet validated on this hardware. Pin weights/dependencies, avoid training from scratch, and start with short clips. A large-model fine-tuning project is outside the MVP.

Select the engine using validation counts and longest-clip latency, not model reputation. If Qwen has installation, OOM or speed problems after **45 minutes of focused setup**, preserve the CPU baseline and continue integration. If it runs but overflags normal clips, tighten the visible predicates and validate them. Do not simply maximize recall by flagging every clip.

Later, if justified by errors: increase sampling near a proposed event; compare full view and a modest crop; cache the model in a reusable worker; or test a larger model on Brev. Keep these as interchangeable improvements behind the same contract. None is required to start the application.

### Candidate rules, not assumed dataset labels

The official text was checked on 17 October 2026 at `https://adilet.zan.kz/rus/docs/V1900019680`, NBK Resolution No. 231 of 29 November 2019.

| Candidate | Basis | Evidence required | False inference to avoid |
|---|---|---|---|
| Cashier leaves while receipt service is ongoing | Paragraph 22 | Departure plus the same ongoing customer interaction | Standing up or an empty desk after completion |
| Previously accepted cash remains exposed during the next transaction | Paragraph 20 | Visible transaction change and continuity of the earlier cash | Any cash on a desk |
| Receipt completed without visible counting | Paragraphs 19(2), 20 | Entire receipt-to-storage sequence with counter/hands visible | Treating occlusion or sparse sampling as omission |
| Collection bag accepted without a visible check | Paragraph 24(3)–(4) | Complete receipt including the bag/seal and inspection opportunity | Inferring no inspection from a brief clip |

The first two are enabled starting hypotheses. The last two are disabled because absence-based findings need stronger temporal coverage. Determine actual category relevance from the dataset. Paragraph 32 contains payout counting exceptions; never turn “no counting seen” into a universal rule. Do not infer accounting correctness, amount mismatches, intent/theft, signature authenticity or completion in bank systems from these videos. Evidence supports a human reviewer, not a disciplinary verdict.

## Schedule with equal workload

Every row assigns all three people the same elapsed time. Target 12 productive hours each (36 person-hours). Breaks and sleep sit outside these blocks within the nominal 24 hours. Use a countdown to the official **12:00 deadline**; confirm the date and submission cutoff with the organizers. Do not interpret this as permission to work past their deadline.

| Productive time | You — backend and deployment | Ratmir — detection and evaluation | Amir — frontend, evidence and pitch | Shared gate |
|---|---|---|---|---|
| 0–0.5 h | Create shared repo; run API demo | Verify GPU/driver and demo | Run UI; inspect expected result | Agree schema, ownership and 3 acceptance clips |
| 0.5–2 h | Prepare host/key/tunnel path; label one third of videos | Download weights and baseline; label one third | Verify source conditions; label one third | Dataset inventory, split groups, first public synthetic smoke test |
| 2–5 h | Harden API limits/status/errors; benchmark real uploads | Train baseline; run Qwen on short real clips; map observed classes | Wire every result/error state; implement source/evidence layout | **First real MVP**: positive + negative + multiple-event flow |
| 5–6 h | Merge; run remote API test | Freeze candidate model/config | Check evidence/timestamps against clips | **MVP gate**: full flow, no manual code launches |
| 6–8 h | Longest MOV; cold starts; restart/timeout/offline tests | Validation confusion counts and per-type errors | Inspect all false positives locally; check source applicability and new-angle UI | One known-good release; prioritized failures |
| 8–10 h | Resolve reliability bottleneck; stable public endpoint | Fix largest measured detection error; compare on validation | Improve evidence clarity; prepare six-slide outline with measured results | **Feature freeze**; no new product features |
| 10–11 h | Final unseen test via public API; submission instructions | Final model hash, limitations, reproducibility | Record backup demo; finish pitch visuals | Submission bundle complete |
| 11–12 h | Submit URL/API/key; check another network; monitor host | Explain ML choices; answer failure questions; verify fallback | Lead two timed rehearsals; shorten to five and two minutes | Submission verified before cutoff |

If a task finishes early, help with the shared error review or test another machine; do not expand that person's feature list. If Ratmir is behind on labels, you and Amir take the remaining clips. Keep his model-development time protected.

If **16 hours** are truly available, preserve the last two hours and insert four hours before freeze: 1 hour of targeted dataset cleanup (all three), 1.5 hours for one model/sampling experiment (Ratmir runs it, you measure resources/latency, Amir independently reviews evidence), 1 hour of stress/angle tests (each person owns a different set), and 0.5 hour of integration. Cut the experiment immediately if it does not improve held-out results. Do not spend the extension adding unrelated features.

## Data and evaluation procedure

1. Download to a local ignored folder. Record filenames, duration, FPS/codec and organizer label/timestamp without displaying private video content to cloud AI. The brief describes at least 40 staged, silent, overhead videos, 10–145 seconds, around 29 MB; inspect actual files rather than assuming these statistics.
2. Each person reviews roughly one third locally in a normal video player. Add visible action type, interval, event timestamp, ambiguity and applicable rule. Agree on labels together. A positive timestamp alone does not make all other frames negative.
3. Group near-duplicates by recording session, actors and camera. Aim for about 60/20/20 train/validation/test if class counts allow (roughly 24/8/8 for 40 clips). Never split frames of one clip across sets. With scarce classes, prioritize a defensible grouped split and report the counts.
4. Train on train only. Tune thresholds/prompt/window settings on validation. Open the held-out test once after freeze. Do not use jury test videos to train or tune.
5. Report `TP/TN/FP/FN`, processed/failed count, precision/recall, type errors, matched time errors, p95/max cold public latency, and sample counts. The evaluator's three-second tolerance is an internal diagnostic, **not** the official criterion. The brief asks for a concrete frame/time but gives no numeric tolerance.
6. Manually verify every false positive and false negative. Count missed unsupported categories separately from localization errors. Do not claim zero misses generally from a tiny test set.
7. Try an actually different angle if available; a crop/rotation is a stress test, not proof of camera generalization. Compare semantic reasoning with the reference baseline's background sensitivity.

### Acceptance checks before submission

- From another network, upload a fresh 29 MB MOV with the jury key, poll and get valid JSON in under 300 seconds. Internal target: **under 240 seconds** to leave room for transfer and variability.
- Positive, negative and multiple-event clips behave sensibly; each event has 1–3 JPEG evidence frames, a frame timestamp, type and visible explanation.
- Wrong/missing keys are rejected on both endpoints. Corrupt/oversized/overlong videos and model failures never return a clean verdict.
- All enabled classes have a reviewed rule and a tested example. Any unsupported category is stated in limitations.
- Run offline once after downloading weights. Reconnect and verify the public URL afterward.
- Reboot/restart the service in a rehearsal, ensure the model and key are set, and test the actual launch command.
- Check the laptop power/sleep/network state, enough disk, full-event availability and deadline margin.
- Ensure the jury runner's concurrency and tolerance for additional JSON fields/HTTP 429 are understood. This starter serializes jobs and rejects overload, rather than queuing unbounded work.

## Brev credit decision

Your prior use to rent a GPU and train strongly suggests **Brev compute credit**, not hosted inference credit. Confirm the organization balance and quoted total hourly cost in the console; an API key by itself is a credential, not a dollar balance.

Use the RTX 4060 for the first MVP. Spend **$0 initially**. If the laptop cannot meet the measured quality or latency target, allocate at most **$10–15** to one short controlled larger-GPU experiment, keeping the remainder for a backup or the judging window. Running the full app and model on your own rented VM is architecturally different from calling a hosted AI API, but the brief's wording about “cloud solutions” is broad: ask the organizer whether a team-controlled Brev VM and data transport are allowed before uploading event data there. A laptop-hosted service avoids the GPU-hosting ambiguity.

Compute examples are budgets, not live price quotes: at $1/h, 12 h costs $12; at $1.50/h it costs $18; at $2/h it costs $24, before storage/other charges. Reserve at least $5 and budget through the end of judging, not just coding. Check the actual all-in price and availability before launching. Stop compute when unused, but keep the submission host alive while judges test. Back up code/config before stopping; capacity may be unavailable on restart. Delete only resources/data you intend to remove. Never put the Brev key in Git, slides, browser JS or a prompt.

## Pitch and submission

Amir leads a five-minute pitch; you and Ratmir each own a short evidence-based explanation and Q&A. Suggested timing: 0:00–0:20 one-sentence result and workflow; 0:20–2:00 live example; 2:00–3:00 measured held-out performance and an honest error; 3:00–4:00 why the source/evidence/offline design matters; 4:00–4:40 deployment and limitations; 4:40–5:00 closing demonstration of the exact event-to-rule link. Also rehearse a two-minute version. Do not invent a time-saved percentage.

Prepare six content slots: reviewer task; live finding screenshot; before/during/after evidence; measured results with counts; local deployment/source mapping; limitations and next step. Slides are optional in this case; the live flow and a backup recording matter more. Use only event media in a presentation if the event's sharing rules allow it; use synthetic screenshots for public sharing otherwise.

Submission packet: UI URL, API base URL, private API key, exact startup instructions, architecture/model/license details, supported event types, source catalog, measured results and limitations. Confirm successful receipt. Save a known-good source/model/config snapshot and a short demo recording; retain no jury data beyond allowed use.

## Questions to resolve without blocking unrelated work

- Is a self-hosted model on a team-controlled Brev VM allowed? Are HTTPS tunnels acceptable for transporting uploads?
- Will the API harness send concurrent jobs, retry 429, and allow additional `rule`/`meta` fields?
- Confirm the deadline date, submission channel and whether modified code prepared before the event is permitted under FAQ 24. FAQ 35's reference to “ready-made solutions” should not be interpreted as a ban on every open-source building block, but confirm their intended boundary.
- Obtain the authorized dataset link and actual annotations. No model accuracy claim is possible until it is used locally.
