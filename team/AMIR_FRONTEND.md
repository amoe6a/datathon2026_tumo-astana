# Amir's Codex prompt for frontend and presentation

Open the shared `forte-review` folder in Codex and paste the task below. Obtain the entire code ZIP/repository as well as this file. Leave the common root `AGENTS.md` in place; explicitly reference this role prompt.

---

You are helping Amir own frontend, evidence presentation and the pitch for a three-person ForteBank Datathon team. Other teammates own backend/deployment and ML. We each have 12 productive hours. Build on the existing UI and make the required workflow understandable and reliable. Do not spend the first hours building slides while the product has missing states.

Read `AGENTS.md`, `README.md`, `HACKATHON_PLAN.md`, `web/`, the existing API in `app/main.py`, and `rules/catalog.json`. The current UI is a working buildless HTML/CSS/JS app served by FastAPI, not a static mockup. It has a clearly labeled synthetic demo and real API integration. Preserve that simplicity; no frontend framework migration is needed for the MVP.

## Constraints and evidence

The reviewer uploads MOV, sees processing, then either no detected violation or possible issues, each with type, timestamp, explanation, 1–3 source JPEGs and a reviewed RK rule. More than one event is possible. Do not invent a third verdict or show an API/model failure as a successful clean result. Include uncertainty without pretending that an empty list certifies compliance. Synthetic clips and screenshots must remain labeled synthetic.

No face identification, audio analysis, invented legal links, unsupported monetary impact or unmeasured accuracy percentages. No new login/account/history/chatbot/dashboard features before the core gate. The API uses an access key; keep it only in browser memory and out of source, screenshots, slides and Git.

Do not send event videos, frames or private frame descriptions to cloud AI, including Codex. Team members review bank footage manually in a local player. Codex can inspect code and synthetic UI examples; private footage is not to be rendered into tool/model outputs. Report aggregate defects and test states without revealing frame contents. Public portfolio screenshots should use the synthetic fixture unless the organizer explicitly permits event footage sharing.

## Setup and ownership

Use Python 3.11 and a project `.venv`; no Node dependency is needed for this UI. Run the synthetic service locally from README or `run.ps1 -Mode demo`. Do not install GPU dependencies on my laptop unless it is the shared RTX host.

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m scripts.make_demo
.venv\Scripts\python.exe -m pytest tests -q
```

Run the app, enter the local generated access key, and try Normal and Two events. Verify actual page state after actions. Use ordinary local preview tooling; do not replace the backend with hardcoded result JSON or a cloud AI endpoint. Reuse the current visual style unless there is a concrete usability reason to change it.

Own `web/`, `docs/source-review.md`, `docs/pitch.md`, presentation assets and `docs/frontend-handoff.md`. Ratmir owns rule catalog edits; send him verified source findings. The backend teammate owns API and deployment. Use branch `frontend` in a separate clone. Keep the existing routes/response shape; request contract changes instead of silently breaking them. Do not force-push.

## My 12-hour work queue

1. **0–0.5 h:** Run the UI and synthetic examples; understand every output field. Agree the three acceptance videos and source display with teammates.
2. **0.5–2 h:** Label my third of the dataset manually. Audit the official RK source conditions for observed categories. Create a local checklist of what each rule requires and excludes.
3. **2–5 h:** Finish upload, progress, positive, negative, multi-event, missing key, busy, timeout and failed states. Ensure each event has clear source frames, timestamp controls and rule text. Test small screens and keyboard operation. Do not introduce dependencies/CDNs that break offline use.
4. **5–6 h:** Integrate real API outputs with teammates; validate that timecodes/images belong to the supplied clip. Ensure no placeholder stats remain. Complete the real MVP gate.
5. **6–8 h:** Review false positives locally with Ratmir, assess evidence relevance and legal applicability, test bad/slow uploads and unsupported browser codecs. Evidence images must remain usable even when a MOV cannot play in the browser.
6. **8–10 h:** Improve only clarity and reliability. Draft six pitch content slots using actual product screenshots and verified aggregate metrics. Leave results blank until measured, never populate optimistic placeholder percentages.
7. **10–11 h:** Record a short backup demonstration with no secrets; finish five-minute and two-minute narration, sources and limitations. Coordinate permitted footage use with event rules.
8. **11–12 h:** Lead two timed rehearsals, assign transitions to teammates, and verify the live demonstration on the submission URL. Keep a backup video ready for network failure.

## API agreement

`POST /api/v1/analyze`, multipart `file`, header `X-API-Key`, returns `job_id`/processing. Poll `GET /api/v1/jobs/{job_id}`. On done: display `verdict`, `violations`, `meta`. Each violation contains `type`, `timecode_sec`, `description`, `evidence_frames`; the existing implementation adds `rule`, `observation`, optional interval/score fields. On failed: show error, allow retry and do not render a clean verdict. On 429: explain that one job is processing and allow a later retry. Do not silently loop unlimited submissions.

Render filenames, model observations and rule text as text, not injected HTML. Use source URL links safely. Do not label baseline scores as probabilities. Keep timestamps in seconds internally and format for display. Give images useful alt text and ensure keyboard focus is visible.

## Rule audit

Use the official current text at `https://adilet.zan.kz/rus/docs/V1900019680`. Relevant verified starting sections: paragraph 22 (departure before completion); paragraph 20 (customer presence/storage of previously received cash); 19(2) (receipt counting); 24(3)–(4) (bag integrity/seal); 32 (payout counting exceptions). These are research starting points, not a complete known taxonomy.

For each class Ratmir proposes, record document title, paragraph, official URL, effective-version note, visible precondition, exceptions, what footage cannot establish and one false-positive scenario. An isolated desk with cash is not enough to establish paragraph 20. No counting seen in sampled frames is not proof that counting did not happen. Keep unsupported categories disabled. Do not make legal claims beyond the visible evidence.

## Pitch structure

The case specifies five minutes plus two minutes Q&A, with presentation optional. Prepare a two-minute compressed version as backup. Lead with what the system found and show the reviewer clicking a timecode and checking the corresponding source rule. Use about 20 seconds for the problem; the judges already know it.

Six content slots: reviewer task; live result; before/during/after evidence; held-out results with sample counts; offline/source-mapping approach; limits and next step. Aim for 0:20–2:00 live flow, 2:00–3:00 measured results, 3:00–4:00 design reasoning, final minute limitations and conclusion. No fabricated time saved, cost saved or accuracy. If no evaluation is finished, state that plainly and prioritize obtaining it rather than polishing a stack slide.

My teammates each speak briefly: backend owner explains reproducibility/API timing; Ratmir explains the validation and biggest failure. Prepare Q&A on camera changes, unknown events, false positives, local operation, the rule mapping and model licensing. Do not claim that synthetic demo success is evidence of bank-video detection performance.

## Definition of done

Deliver a UI verified against the live local API, readable source/evidence states, an audited source note, a five-minute and two-minute script, permitted backup demo and final handoff. `docs/frontend-handoff.md` lists pages/states tested, screens checked, unresolved usability issues, commit and next action. It contains no secrets or private event images. Do not declare readiness after viewing only the empty upload screen.

Begin by running the existing app and testing both synthetic examples. Improve observed gaps before adding anything new.
