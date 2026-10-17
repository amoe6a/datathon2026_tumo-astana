# Shared project instructions

This repository is a ForteBank Datathon prototype. Read README.md and HACKATHON_PLAN.md, then the role file explicitly assigned by the human. Do not assume a role based on which file happens to exist.

- Preserve one complete workflow: upload, local analysis, binary verdict, event time, JPEG evidence and reviewed RK rule.
- All inference stays local; download weights only during setup. Never transmit event videos, frames, descriptions or embeddings to cloud AI, including the coding assistant. Run local code with aggregate-only diagnostics; humans inspect event footage themselves. Synthetic fixtures are safe for model-facing debugging.
- No face identification or audio. No claims about invisible accounting data, intent or legal guilt.
- No invented metrics, sources or event types. Distinguish implemented/tested/synthetic/unverified states.
- Root API: POST /api/v1/analyze with multipart file; GET /api/v1/jobs/{job_id}; X-API-Key required. Preserve exact verdict and status enums.
- Keep demo weights separate and rejected in competition mode. A failure must not return no_violation.
- Backend owner: app/, API tests, deployment and shared integration. Ratmir: ml/, model/evaluation and rules/catalog.json. Amir: web/, source audit and presentation. Coordinate shared files.
- Core commands: python -m scripts.make_demo; python -m pytest tests -q; python -m scripts.doctor. Use a project Python 3.11 virtual environment and the installed interpreter.
- Keep secrets, runtime evidence, weights and videos out of Git. Do not print credentials or private observations to the coding chat.
- Do not rewrite teammate work or force-push. Use small commits and explicit handoffs.
- Finish the MVP before adding features. No extra accounts, dashboard, chatbot, vector database or orchestration system without a measured need.
- Run targeted meaningful tests; report what ran and any remaining limits. GPU and public deployment success require actual checks, not written instructions.
