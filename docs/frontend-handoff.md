# Forte Review frontend handoff

Implemented and checked on 17 October 2026 in the original extracted project.
Prepared for publication in a separate fresh clone of the shared repository,
on branch `frontend`, based on `main` at `56eb7e9`. See the branch's Git history
for the final commit; no backend or ML changes are part of this handoff.

## Implementation

The blank `web/` now contains the complete Russian-language upload, local
analysis and evidence-review interface. It is buildless HTML/CSS/JS served by
the existing FastAPI `/` and `/assets` routes. Backend, ML, rules and dependency
manifests were not changed.

The layout combines the explicitly selected Brainwave and AI SaaS Landing
references. Exact inspected source files, revisions, the MIT notice, and the
Brainwave license limitation are recorded in `web/THIRD_PARTY_NOTICES.md`.
Brainwave code and assets were not copied. The AI SaaS template's layered
conic-gradient border was adapted; its React and cloud dependencies were not
imported. Orbital geometry and other decorative artwork are local CSS/SVG.
System fonts support Cyrillic without downloading fonts or contacting a CDN.

The API key stays in the password field/browser memory. It is not stored in
web storage, URLs, source or application logs. Demo buttons are shown only
after `/health` reports `demo_only`. Source catalog data comes from the
authenticated `/api/v1/rules` endpoint, including enabled/disabled state.
Catalog entries and model observations are shown in their original language
to preserve the backend's meaning; current catalog text is English.

Manual status recovery polls the accepted job without re-uploading. Polling
has a five-minute budget from the initial action; manual status recovery has
its own bounded window and cannot change the backend job deadline. HTTP 429
honors Retry-After without automatic resubmission. Failed/malformed responses
do not display a clean verdict. Optional metadata/rules have explicit fallbacks.

## Run locally

From the project root in PowerShell:

```powershell
.\run.ps1 -Mode demo
```

Open `http://127.0.0.1:8000`. Use your locally generated session key, then
choose Normal or Two events. The script prints its temporary key in your own
terminal. Never put the jury key in this handoff or screenshots.

For verification, an isolated Python 3.11.17 environment was installed in
the original working copy's `.venv` with the dependencies already pinned by
`requirements.txt`. Its managed runtime is under that copy's ignored
`runtime/frontend-setup/`. These environment files are not published.
On a fresh clone use the README's Python 3.11 setup. No Node.js installation
is needed to run the application.

The temporary preview was run on `127.0.0.1:8765` in demo mode with an isolated
`runtime/frontend-preview` directory and the existing automated-test credential,
not a jury or production key. This localhost-only preview is for synthetic data.
It is not a public deployment.

## Executed verification

- `node --check web/app.js`: passed.
- `node --test tests/frontend.test.cjs`: 11 passed. Covers missing key/file,
  unsupported/empty/oversized uploads, 401, 413, 415, 429 cooldown, 503, network
  failure and recovery without duplicate upload, five-minute polling budget,
  failed/malformed/contradictory results, negative verdict, plain-text rendering,
  disallowed source URL protocols, and frame dialog open/close. These are
  browser-independent tests with controlled responses, not real inference.
- `.venv/Scripts/python.exe -m pytest tests -q --basetemp
  runtime/frontend-pytest-approved`: 18 passed in 38.02 seconds. One existing
  Starlette/httpx deprecation warning; no failed checks.
- Live browser, actual FastAPI + spawned local worker: Normal returned
  `no_violation`; Two events returned `violation` with two events and six
  successfully decoded source JPEGs. This measures only the synthetic workflow.
- Real file chooser: a synthetic corrupt MOV produced `failed`, a visible error,
  enabled retry and no clean result. Unsupported TXT was rejected before upload.
- Wrong API key produced a field error with focus on the password input.
- Real catalog request displayed four non-demo rules and their enabled/disabled
  state, conditions and official-source links from the returned catalog.
- Frame enlargement, Escape dismissal with focus restoration, event selection,
  and visible keyboard focus on the Analyze button were checked.
- Initial and result layouts checked at desktop/tablet/mobile widths:
  1440, 1280, 1024, 768, 390 and 375 CSS pixels. Document scroll width did not
  exceed its client width after the responsive fixes.
- Browser console inspection showed no JavaScript errors during the checked
  synthetic flows. No inline scripts, inline handlers or external runtime assets.
- Reduced-motion CSS disables animation, transitions and smooth scrolling.
  The OS/browser reduced-motion preference itself was not toggled during testing.

Screenshots in `docs/frontend-preview/` use only the empty interface or generated
synthetic scenes. No private bank footage or actual API key is included.

## Remaining limits and next action

The generated MPEG-4 MOV codec was unsupported by the tested browser. The
fallback kept all six evidence frames usable. Actual seeking on a browser-playable
H.264/HEVC recording, drag-and-drop with the OS pointer, and other browsers still
need a manual check. File selection through the real chooser was verified.

Real event-data quality, GPU inference, public URL access and public five-minute
timing were not verified. Frontend error states for busy/timeout/network loss
were exercised with controlled responses; the backend's corresponding real
admission/timeout behavior was covered by the existing API tests.

Next: integrate the team's measured model, confirm actual MOV variants, then
check positive/negative/multiple-event recordings and the deployed URL together.
