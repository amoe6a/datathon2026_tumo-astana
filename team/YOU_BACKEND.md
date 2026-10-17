# Your Codex prompt for backend and deployment

Copy the text below into Codex after opening the shared `forte-review` project folder. This file is a task prompt, not a file that Codex automatically loads by its name. Keep the common `AGENTS.md` at repository root. Give teammates the same repository/ZIP so everyone extends the existing implementation.

---

You are helping me own backend and deployment for a three-person ForteBank Datathon team. Ratmir owns ML; Amir owns frontend and presentation. Treat us as equal contributors with 12 productive hours each. My strengths are backend and DevOps. Work on my portion, integrate teammates through explicit contracts, and do not rebuild their components.

Read `AGENTS.md`, `README.md`, `HACKATHON_PLAN.md`, `app/`, `ml/worker.py`, and `tests/test_pipeline.py`. Inspect the actual current files before proposing changes. A working synthetic application already exists. Your first task is to run it, find real failures and prepare deployment. The real dataset is not included; ask me only for missing local paths/access when required, while progressing on independent work.

## Non-negotiable case requirements

- UI uploads MOV; API `POST /api/v1/analyze` with multipart `file`, returning `job_id`; `GET /api/v1/jobs/{job_id}` with `processing/done/failed`.
- Both endpoints require `X-API-Key`. Final `verdict` is exactly `violation` or `no_violation`. Each event has `type`, numeric `timecode_sec`, `description`, and 1–3 `evidence_frames` with timestamp and JPEG base64.
- The jury requires a public URL and a response within five minutes per video. Target under 240 seconds end-to-end for the longest case clip. Do not equate returning a job ID quickly with finishing on time.
- All model execution stays within our service. No event videos, frames, frame descriptions, transcripts or embeddings may be sent to cloud AI APIs, including Codex. You may edit code and run local analysis, but return only non-sensitive aggregate diagnostics to the chat. Do not open event frames with a model-facing image tool or print per-event descriptions. Team members inspect private footage themselves in a local player.
- No face identification, audio processing, invented legal citations or claims about invisible bank-system data. Preserve synthetic-demo labeling and reject demo models in competition mode.

## Environment preparation

The primary inference host is a team laptop with RTX 4060 8 GB, 32 GB RAM and i9-14900HX. My current coding laptop may differ. Inspect Python, ports, Git and hardware without exposing secrets. Use Python 3.11 with a project `.venv`; no Node is required. Do not change system security settings or install heavy GPU packages on every teammate's computer.

Run:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m scripts.make_demo
.venv\Scripts\python.exe -m pytest tests -q
.venv\Scripts\python.exe -m scripts.doctor
```

Reuse an existing environment. If Python's launcher is missing, locate a Python 3.11 executable instead of repeatedly failing. Use `run.ps1 -Mode demo` or README's explicit commands to start. Generate the API key locally; show it only to me in the local terminal, never add it to tracked files or frontend JavaScript. Use a distinct jury key for public deployment. Run one Uvicorn worker unless admission control is made shared.

## Ownership and interfaces

I own `app/`, `Dockerfile`, runtime configuration, startup scripts, API tests, deployment instructions and `docs/backend-handoff.md`. Ratmir owns `ml/` and `rules/catalog.json`; request a contract-compatible change instead of rewriting them. Amir owns `web/` and presentation notes. Coordinate changes to `requirements.txt`, root README and shared contracts through me as integrator.

The inference contract is `ml.engine.analyze(path, model, engine, rule_path, progress)` returning `verdict`, `violations`, `meta`. `ml.worker` writes structured result/progress files and never writes private footage to logs. The UI consumes the existing routes. Preserve unknown-but-needed extra rule/meta fields; check whether the jury accepts them.

Use branch `backend` in my separate clone. Make small commits; do not force-push or reset teammate work. I merge Ratmir's `ml` and Amir's `frontend` branches at the shared checkpoints. If there is no shared Git remote yet, help me initialize the local repository and explain how to add the team remote; do not invent its URL.

## My 12-hour work queue

1. **0–0.5 h:** Run the demo and tests; freeze routes/schema with teammates; create the repo and integration checklist.
2. **0.5–2 h:** Establish the deployment path and label one third of the clips manually. Check the event's interpretation of a public tunnel/team-controlled Brev VM. Create the first public synthetic smoke test if permitted. Preserve the key privately.
3. **2–5 h:** Test authentication before upload parsing, file/duration limits, corrupt MOVs, busy responses, missing model, interrupted jobs, persistence, worker timeout and raw-file deletion. Fix actual failures. Benchmark a real video through the integrated API without returning its content to chat.
4. **5–6 h:** Merge both branches; prove positive, negative and multi-event workflows through the public URL. Ensure there is no manual per-video code launch.
5. **6–8 h:** Run the longest real MOV cold; measure total response time. Rehearse restart and offline inference. Ask whether the jury sends concurrent requests; the starter rejects overload with 429. Add queueing/capacity only if necessary and measured within deadline.
6. **8–10 h:** Fix the largest reliability issue, stabilize the public URL, document exact startup/model configuration and freeze the release. Do not add Redis/Celery/Kubernetes just for presentation.
7. **10–11 h:** Run unseen final tests from another network; prepare submission URL, key, source/model hashes, startup instructions and fallback release. Keep the host plugged in and awake.
8. **11–12 h:** Submit before the deadline, verify receipt, keep the host available and rehearse my short architecture/deployment explanation with the team.

## Deployment decision

Prefer the RTX laptop for inference. A tunnel forwards HTTPS to localhost; it is not an inference API, but data transport must be acceptable to the organizers. A stable named tunnel is preferable. Quick tunnel command after `cloudflared` is installed from its official source:

```powershell
cloudflared tunnel --url http://127.0.0.1:8000
```

Do not stop at seeing a URL: run `scripts.smoke_api` from a second network with a real authorized test upload and verify completion. Test the jury's exact route/key/field. Keep the URL alive through judging. The CPU Dockerfile does not install CUDA dependencies; do not present it as a tested GPU container.

Brev credit is reserve. Do not create paid resources until we choose an all-in hourly rate and budget. If the organizer permits our own GPU VM, keep the full application/model together there and preserve offline portability. Never replace the local model with a hosted NVIDIA/GPT/Gemini inference endpoint.

## Definition of done and handoff

Deliver a runnable release, passing relevant tests, remote smoke evidence with counts/timings only, exact startup command, known limits and a private submission checklist. `docs/backend-handoff.md` must state commit, environment, modes tested, endpoint status, longest cold latency, concurrency policy, remaining blocker and next action. Do not include the key, bank media or event descriptions in this document. Report what actually ran; do not claim a public deployment or GPU check merely because instructions exist.

Start now with environment inspection and the synthetic test. Ask targeted questions only when a concrete missing input blocks the next action; continue useful independent work meanwhile.
