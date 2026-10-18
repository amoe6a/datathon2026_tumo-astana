# Local development

Use a separate checkout and runtime for development. Do not restart the judge service, run GPU experiments or reuse its key/runtime during judging. Deployment and recovery are documented in [Linux deployment](linux-deployment.md).

## Install and test

Core development and synthetic tests use **Python 3.11**:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m scripts.make_demo
.venv/bin/python -m pytest tests -q
.venv/bin/python -m scripts.doctor
```

Optional frontend checks require Node.js; the application itself does not:

```bash
node --check web/app.js
node --test tests/frontend.test.cjs
```

On Windows, use `py -3.11 -m venv .venv` and `.venv\Scripts\python.exe` in place of the Linux interpreter path.

## Synthetic application

Synthetic fixtures test transport, evidence and failure handling. They do **not** demonstrate real detection quality.

For a Linux development checkout, first set `FORTE_API_KEY` to a separate, locally generated development key; do not commit it or reuse the judge key. Then:

```bash
export FORTE_ENGINE=demo
export FORTE_MODEL=models/demo.npz
export FORTE_RUNTIME=runtime/local-demo
.venv/bin/python -m uvicorn app.main:create_app --factory --host 127.0.0.1 --port 8001 --workers 1
```

Open `http://127.0.0.1:8001`, enter the development key and use the synthetic demo buttons. If the port is already occupied, choose another port; do not stop an existing service.

Windows also supports `.\run.ps1 -Mode demo -Port 8001` from PowerShell. Set `$env:FORTE_RUNTIME` to a separate development directory before starting alongside another service.

## Real model

The selected GPU release uses a separate, already-tested `.venv-release`, whose exact dependencies are recorded in `requirements-linux-qwen-lock.txt`. **Do not upgrade it in place.** The CPU development requirements do not install or reproduce the GPU release.

- [Pinned runtime, model setup and rollback](linux-deployment.md)
- [Training and evaluation commands](qwen-training-deployment.md)
- [Human annotation requirements](ml-labeling.md)
- [Release configuration](../configs/qwen-release.json)

Base weights, the selected private adapter and authorized video data must be supplied separately. Never substitute synthetic weights for the trained release. Competition videos, frames, descriptions, predictions and embeddings must remain local.

## Distribution

Use GitHub's **Code → Download ZIP** for the source tree, or clone the repository. A duplicate generated archive is intentionally not committed. If a standalone source-only handoff is needed, `python -m scripts.build_team_kit` creates one locally under ignored `team-kit/`.

Historical handoffs and aggregate reports record earlier experiments. They do not override the selected release configuration or current deployment instructions.
