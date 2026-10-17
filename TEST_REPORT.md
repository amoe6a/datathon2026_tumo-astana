# Verification report

Verified on 17 October 2026 in a Windows/Python 3.11 virtual environment. The authoring machine has an MX250 2 GB; it is not the team's RTX 4060 host. No event dataset was supplied here.

## Executed checks

18 automated checks passed across the final pipeline and edge-case test runs. The pipeline run completed in 64.38 seconds; the additional edge-case run completed in 1.20 seconds. A dependency deprecation warning about Starlette's HTTPX test client did not fail the checks.

- Actual synthetic MOV upload → spawned worker → polled JSON for normal, one-event and two-event clips.
- Correct verdict/event counts, times within the generated event intervals, 1–3 decodable JPEG frames and an evidence frame matching each reported timestamp.
- Missing/wrong API key, unsupported extension, empty/corrupt video, unknown job, oversized request and oversized uploaded file.
- Busy admission returns 429 with a retry hint; timeout produces failed and releases the slot.
- Interrupted jobs become failed on server restart.
- Missing model does not produce a clean verdict; demo weights are rejected in competition mode.
- Model JSON validation rejects unknown rules, out-of-window, non-finite and boolean timestamps, malformed JSON and invalid uncertainty types.
- Enabled real rule entries have official source URLs and observable conditions/cautions.

Additional checks: JavaScript syntax; Python compilation; live HTTP smoke test of a 10-second, two-event synthetic MOV. The observed API end-to-end latency in that run was **3.81 seconds**. This is a functional measurement on a toy fixture, not a bank-video performance result or prediction for a 145-second clip.

## Browser verification

The local interface was opened and its actual synthetic upload, progress and multiple-event results were checked. It displayed two events, source timestamps, three source frames for each event, rule/test-condition text and the synthetic disclaimer. Browser playback of the generated MPEG-4 MOV was unsupported; the application hides the failed player and retains usable evidence frames. The normal result was also checked. Screenshots in the delivery show synthetic scenes only.

## Frontend integration review — 17 October 2026

PR #1, frontend revision `73adee8`, was reviewed and merged with the existing backend and ML implementation. The reviewed code passed all 11 frontend behavior tests and all 18 Python/API tests (61.34 seconds for the Python run), plus the JavaScript syntax check. The Python run reported one existing Starlette/HTTPX deprecation warning. The merged `app/`, `ml/`, `web/`, `tests/` and `rules/` trees match that tested revision; integration only updates documentation and the team-kit archive in addition to the merge.

The frontend covers upload validation, API errors, Retry-After, bounded polling, recovery without duplicate upload, verdict validation and evidence dialogs. See `docs/frontend-handoff.md` for Amir's separate live-browser checks and remaining browser limitations. These checks use synthetic or controlled responses; real video detection quality and GPU performance remain unverified.

## Not verified

- Detection quality, label coverage or confusion counts on ForteBank event videos.
- Qwen weight download/import/inference, GPU VRAM use and latency on the team's RTX 4060.
- Public URL, remote jury access, actual harness concurrency, or public upload timing.
- Docker build/run, actual H.264/HEVC MOV variants, variable-frame-rate/rotation behavior, or actual new-angle generalization.
- Offline execution of the Qwen engine after download. Offline flags and local-only loading are implemented, but the target test must still run.

Treat the application as a tested software baseline. The competition release requires the remaining data, GPU and deployment checks listed in the plan. No model accuracy percentage is claimed.
