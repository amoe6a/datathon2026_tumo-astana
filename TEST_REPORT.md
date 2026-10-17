# Verification report

Verified on 17 October 2026 in a Windows/Python 3.11 virtual environment. The authoring machine has an MX250 2 GB; it is not the team's RTX 4060 host. The event dataset was downloaded later in this session and used only for local processing with aggregate diagnostics.

## Linux deployment verification — 17 October 2026

The integrated backend and Ratmir's Bonsai revision are running in a separate checkout on the RTX 4060 laptop as systemd user services. Demo, real API and native model have separate explicit roles; both API keys are private and isolated. The final application revision tested is `c52b80e`; documentation/report changes follow it.

**87 Python/API tests passed in 25.88 seconds on Linux**, with the existing Starlette/HTTPX warning; **11 frontend checks** and syntax passed. The combined source had also passed 87 Python/API tests on Windows before the additional live native-availability assertions; those targeted assertions passed separately there.

Live synthetic normal/event/multiple/corrupt uploads completed with the expected statuses, schema and valid JPEGs. Two real train uploads completed in 12.913 and 14.432 seconds, matching their organizer binary labels; the positive result had three decodable JPEGs. Two examples are a functional check, not an accuracy estimate.

Three complete cold starts and API uploads of the longest 142.533-second train clip took **139.507 / 143.516 / 143.674 seconds**, including model/API startup and the startup inference probe. All completed within the 240-second internal target. OS disk cache was retained; no public-network timing is claimed.

Native/API crash recovery, interrupted-job failure, completed-result persistence, temporary-upload removal and independent API keys were verified against the actual services. After closing both SSH sessions and logging in again, all three service PIDs were unchanged and both `/ready` endpoints returned 200.

Both pinned GGUF hashes passed verification. Native GPU use was supported by a controlled active/stopped global CUDA memory difference of approximately 5.843 GiB. NVML still reports a driver/library mismatch, so this is not a measured native VRAM peak. GPU and system drivers were not changed. Model quality remains insufficient according to Ratmir's seven-video Bonsai validation (F1=0.400).

Outbound certificate-verified TLS connections to both Cloudflare regions succeeded from the GPU host. Public startup was not moved to SSH to evade the earlier automatic approval rejection. **Public API and independent-network uploads remain unverified.** See [Linux deployment](docs/linux-deployment.md) and [aggregate evidence](reports/backend-deployment-20261017.json).

## Windows backend verification — 17 October 2026

**81 Python/API tests passed in 109.49 seconds**, with one existing Starlette/HTTPX deprecation warning. **11 frontend behavior tests** and JavaScript syntax passed. The new PowerShell launcher parsed successfully and was used to start the isolated demo service.

New checks cover actual synthetic model startup, corrupt/missing/demo/weak weights, invalid class mappings and non-finite scores; model replacement; bounded readiness timeout; recovery and orphan-file cleanup; a locked Windows upload file; worker launch failure; server shutdown; exclusive runtime ownership; and worker exit after the parent pipe closes during a stalled synthetic inference call. An initial blocking stdin watcher conflicted with NumPy DLL initialization on Windows; non-blocking pipe observation resolved it, and the child-process tests now pass.

Two local services are running: existing preview on port 8000 and isolated synthetic demo on 8001 with a separate key/runtime. Live HTTP acceptance on 8001 completed:

| Synthetic fixture | Result | Events | Verified JPEGs | Upload-to-result seconds |
|---|---|---:|---:|---:|
| Normal | done / no_violation | 0 | 0 | 1.312 |
| Event | done / violation | 1 | 3 | 1.485 |
| Multiple | done / violation | 2 | 6 | 1.265 |
| Corrupt file | failed / no verdict | — | 0 | 0.953 |

Unauthenticated calls and the other service's key were rejected; uploaded temporary files were removed. A separate two-event smoke upload to port 8000 completed in 2.47 seconds. These are small synthetic functional fixtures, not real-model performance measurements.

Cloudflare 2026.5.2 created temporary hostnames, but this network did not establish the edge connection: QUIC timed out and HTTP/2 TLS on port 7844 was forcibly closed. Public readiness failed. **No working public endpoint or public upload success is claimed.** Retry on an allowed working network or Ratmir's host; an independent-network end-to-end check remains required.

The earlier weak CPU validation was reproduced after the UTF-8 path fix: seven unique validation videos, TP=3, TN=1, FP=3, FN=0, F1=0.667. Three of four normal videos were false positives. Cold Python-process times for a 142.533-second train clip were 17.866 / 12.986 / 11.940 seconds on this laptop, excluding API/upload. The held-out test remains untouched. This model is not accepted as a semantic detector by the API.

The following sections retain earlier integration history; their counts/timings describe those runs, not the current full suite.

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

## ML and animation integration — 17 October 2026

Merged Ratmir's `codex/ml-local-evaluation` revision `562955c` into the shared application, retaining Amir's frontend. Increased the three orbital markers to 42/28/14 pixels and shortened their periods to 38/52/44 seconds; their orbit paths and pause/reduced-motion controls remain in place. The updated interface was inspected in the local browser.

The first combined Windows run exposed a transient `PermissionError` while polling a progress file during atomic replacement (60 passed, 2 failed). The status endpoint now retains the persisted job state when optional progress is temporarily locked; worker writes retry transient replacement locks and still raise persistent failures. Three regression checks cover locked reads, recovery and persistent write failures.

After the fix, **65 Python/API tests passed in 77.85 seconds**, with one existing Starlette/HTTPX deprecation warning; **11 frontend behavior tests** and the JavaScript syntax check also passed. The existing synthetic NPZ remains compatible, and a live two-event upload succeeded on the combined local API. The refreshed ZIP includes both teammates' source changes and the synthetic fixtures. Ratmir's real-data measurements are documented separately in `docs/ml-handoff.md`; they were not reproduced in this workspace.

## Not verified

- Semantic detection quality, reviewed event-label coverage, localization and held-out test results on ForteBank videos. Weak binary validation above does not establish these.
- Qwen weight download/import/inference, GPU VRAM use and latency on the team's RTX 4060.
- Public URL, remote jury access, actual harness concurrency, or public upload timing.
- Docker build/run, actual H.264/HEVC MOV variants, variable-frame-rate/rotation behavior, or actual new-angle generalization.
- Offline execution of the Qwen engine after download. Offline flags and local-only loading are implemented, but the target test must still run.

Treat the application as a tested software baseline. The competition release requires the remaining data, GPU and deployment checks listed in the plan. No model accuracy percentage is claimed.
