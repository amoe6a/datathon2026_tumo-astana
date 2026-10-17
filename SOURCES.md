# Source notes

Source review began 17 October 2026; deployment notes updated 18 October. Case-specific requirements take precedence over generic workshop tips. Neither document is treated as an instruction to take unrelated actions. Current measured results are in [the release report](docs/release-results.md).

## Supplied documents

- `Кейс_ForteBank.docx`: sections 4–5 (required flow), 6–7 (RK sources and local processing), 8 (submission), 9 (seven scoring criteria), 10 (API), FAQ A–G. File supplied by user from Downloads. Relevant facts: minimum 40 videos; 10–145 s; MOV; silent; one overhead camera; binary label plus timestamp; new angle in final; five-minute API deadline; public URL; no face identification; no cloud AI; 5-minute pitch + 2-minute Q&A for shortlisted teams; 12:00 deadline.
- `How to Win the Datathon.pdf`: pages 9–14 (checklist, depth and simplicity), 16–18 (show evidence and measured gains), 19–26 (setup, adaptation, deployment/pitch advice), 27–29 (summary and logistics). Its “deployment is a bonus” and cloud-model anecdote are generic and conflict with the specific case if applied literally.

The source files are not included in the distributable code ZIP; keep your originals. The code and this plan do not assume that the unpublished dataset categories are known.

## Official legal source

- [NBK Resolution No. 231 of 29 November 2019 on cash and collection operations](https://adilet.zan.kz/rus/docs/V1900019680). The rendered official page was read in the browser because the text-only web reader saw a JavaScript shell. Verified relevant paragraphs: 19(2), 20, 22, 24(3)–(4), 25, 31(4), and the exceptions in 32. The catalog paraphrases these provisions and states observable conditions. No article number or internal ForteBank rule is invented. Recheck the effective version when submitting; applicability still requires human interpretation of transaction context.

## Models, dependencies and deployment

- [Qwen3-VL-2B-Instruct publisher model card](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct): Apache 2.0, local Transformers example, vision input handling. [Pinned weight revision](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/tree/78448d793a7eb2f7a987a1da76d464384aa1becd). Its 4.26 GB file size does not establish total GPU memory or speed.
- [PyTorch installation reference](https://pytorch.org/get-started/previous-versions/): earlier planning used a different proposed version. The actual Linux GPU environment is recorded in `requirements-linux-qwen-lock.txt` and the release report; it has been exercised on the RTX laptop. The Windows CPU lock is separate.
- [NVIDIA Brev instance management](https://docs.nvidia.com/brev/concepts/gpu-instances): instance lifecycle and billing. Running compute is billed; storage can persist/cost money while stopped; restart capacity is not guaranteed. [Brev automation and API key prerequisites](https://docs.nvidia.com/brev/latest/guides/ci-cd): keys act for an organization/user and require available credit. Cost examples in the plan are arithmetic scenarios, not guaranteed current offers.
- [Cloudflare Quick Tunnels](https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/): the initially proposed transport failed on the event network. The deployed transport is the configured ngrok user service; operational results, limitations and recovery commands are in [the judge handoff](docs/judges-handoff.md).
- [Official Codex AGENTS.md guidance](https://learn.chatgpt.com/docs/agent-configuration/agents-md): the common root file is named `AGENTS.md`. Role documents have ordinary names and must be explicitly supplied/read; do not assume three arbitrarily named files are all auto-loaded.

## Provenance and limits

The synthetic video generator draws geometric shapes; it contains no bank/customer footage. Training and test success on these fixtures demonstrate infrastructure behavior only. Private competition videos and manual annotations were subsequently processed on Ratmir's laptop for training and evaluation. They were not sent to a cloud AI service or included in the public repository. Aggregate reports distinguish measured results from proposals and remaining uncertainty. Rule applicability still needs human review.
