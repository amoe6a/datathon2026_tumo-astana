# Local Qwen training and rule-grounded review

This extension uses the manually annotated workbook and the supplied legal-rule archive locally on the RTX laptop. Event videos, frames, annotations, generated observations and adapter checkpoints stay on that laptop. Only aggregate evaluation reports belong in this repository.

## Data and supervision

The current workbook has 42 video rows and 40 unique video hashes. The earlier frozen split is preserved: 26 train, 7 validation, 7 test after exact-duplicate removal. The test split was previously used for the CPU benchmark; it is not a pristine independent benchmark. Recording-session and camera independence have not been verified. Validation is used for candidate selection; do not tune against the final test results.

`scripts.prepare_qwen_data` matches workbook filenames to actual files, verifies organizer binary labels, preserves frozen splits and keeps the full human descriptions in a private manifest. Its output is not suitable for Git or the coding chat. The `anatation` column supplies actors, actions and violations. It is free text, not reviewed bounding boxes or exhaustive per-class event labels.

`scripts.train_qwen_adapter` performs supervised QLoRA on those human descriptions. It uses only training videos, masks all prompt tokens from the loss, freezes the vision encoder, and adapts language attention query/value projections. The first experiment uses NF4 double quantization, rank 8, alpha 16, dropout .05, learning rate .0001, effective batch size 4, and four epochs. It never uses generated descriptions as ground truth. Lower training loss alone does not establish better detection.

## Inference

1. Decode the uploaded MOV locally, without audio, at two sampled frames per second. Uniformly blank inputs fail.
2. Qwen describes visible roles, objects and actions. The selected release uses whole-clip context; overlapping short windows were tested and rejected for excess false positives. Roles are activity labels, not identities.
3. A local BM25 retriever with English/Russian vocabulary expansion selects relevant cards from the 29-rule catalog. This needs no embedding service or internet connection.
4. Qwen compares actual frames against each candidate's required evidence, exceptions and forbidden inferences. Constrained generation restricts JSON structure and rule IDs; it does **not** guarantee correct visual reasoning.
5. Code rejects ungrounded timestamps, unsupported rule IDs, missing conditions and applicable exceptions. `review_only` cards never produce automatic violations. Accepted findings receive actual decoder frames and fixed official source references.

The trained adapter can be enabled for observations and, in separately evaluated configurations, for visual rule checks. No legal rule is hard-coded as a training class, so retrieval can consider a catalog rule absent from the training videos. This is an architectural capability, not evidence of reliable detection of unseen violation categories. Sparse frames can miss actions; missing sampled actions cannot prove an omission.

## Local commands

The dedicated deployment is `/home/angel/forte-review-service`. Training uses `.venv-training`, which inherits the original CUDA packages through a `.pth` file. The final `.venv-release` contains its own copied packages and does not depend on that original ML folder. The measured runtime is Python 3.14.8, Torch 2.14.0/CUDA 13, Transformers 5.17.0, PEFT 0.21.2 and bitsandbytes 0.50.2. The original CPU application environment remains Python 3.11. This is separate from the older proposed Windows GPU pins in the README.

The additional dependencies are pinned in `requirements-qwen-training.txt`. The pinned LM Format Enforcer revision includes a Transformers 5 tokenizer-import fix; its older PyPI build has the same version number, so an existing installation may require `--force-reinstall --no-deps` for that exact revision. Download dependencies and publisher weights before offline execution.

```bash
.venv-training/bin/python -m scripts.train_qwen_adapter \
  --model models/qwen3-vl-2b --manifest runtime/qwen-training-20261017/dataset.json \
  --output runtime/new-training-run --epochs 4

.venv-training/bin/python -m scripts.evaluate_qwen_rag \
  --manifest runtime/qwen-training-20261017/dataset.json \
  --config configs/qwen-rag-guided.json --adapter runtime/new-training-run/epoch-4 \
  --split val --output runtime/new-validation

.venv-training/bin/python -m scripts.local_review_report \
  --manifest runtime/qwen-training-20261017/dataset.json \
  --evaluation runtime/new-validation
```

Output folders must be new, preventing accidental replacement of prior experiment evidence. Private review HTML contains manual/model descriptions and links to original files; open it only on Ratmir's laptop. Do not upload it to a coding assistant.

The evaluation report distinguishes processed failures from clean verdicts. Binary precision/recall concern organizer per-video labels. The nearest-anchor timing diagnostic is not semantic event matching. Actor accuracy, per-rule accuracy, and different-camera generalization still need independent human assessment.

## Judge API and operation

The original measured baseline is retained below for comparison. The final selected configuration and held-out result are recorded in the release report linked from the README. The trained 2B adapter with BF16 image input completed all seven validation clips: TP=2, TN=3, FP=1, FN=1 (binary F1 0.667), with a 5,334,147,584-byte peak CUDA allocation and at most 18.399 seconds per warm clip. Event timing was poor; the corrected conventional median nearest-anchor error is 31.5 seconds on only two detected positive videos. Earlier reports used the upper middle element (57 seconds) for an even sample count; the aggregate release export corrects that calculation while preserving original reports. Binary agreement is not evidence of correct event type or localization. Multiple configurations have been compared on this small validation set, so selection optimism is substantial. The frozen refined release has now been evaluated once on the seven held-out videos: TP=2, TN=0, FP=3, FN=2, failures=0; F1=0.444 and accuracy=2/7. No tuning used those test results. See [the complete release report](release-results.md).

The constrained decoder requires an explicit Cyrillic alphabet. Without it, Russian observations could terminate inside a string despite staying below the generation limit. Both base and adapter synthetic Russian responses now pass; `scripts.check_qwen_offline` includes this regression check. Invalid output still fails rather than becoming a clean verdict.

Experimental configs compare BF16/NF4 inference, image sequences/native video input, and Russian/English rule prompts. Native video input retains decoder times through millisecond-index metadata and uses temporal frame pairs. `conditional.en.json` translates the 15 conditional rule cards, preserves the number of evidence/exception/caution entries, and is bound to the semantic hash of the Russian source. English prompting did not improve the measured 2B validation result. These candidate configurations are not all recommended for deployment.

The external contract remains `POST /api/v1/analyze`, multipart field `file`, and `GET /api/v1/jobs/{job_id}`, both with `X-API-Key`. Verdicts are `violation` or `no_violation`; failures have status `failed` and no clean verdict. Each violation has a type, description, timecode and one to three JPEG base64 evidence frames. One active job is supported; overload returns HTTP 429 with `Retry-After`.

`scripts.install_service --engine qwen_rag --config <release-config>` creates a managed Linux user service and a matching rule catalog. It explicitly disables keyless local-testing mode. The API key is read from a private runtime file and must not appear in Git or frontend code. GPU processing has a 270-second limit; the public cold-upload path must be measured against the case's five-minute deadline.

`scripts.verify_jury_api` performs real multipart uploads, authentication rejection checks, polling and evidence/schema validation, and saves aggregate diagnostics only. `scripts.check_qwen_offline` blocks socket connections while testing local model load, rule loading and constrained synthetic GPU generation. These are separate from real-video detection evaluation.

Cloudflare's tunnel connection failed on this network. An ngrok agent with traffic inspection disabled is available as the alternative. A connected tunnel alone does not prove the API is ready. Keep Ratmir's laptop and the laptop providing its hotspot plugged in, awake, connected and open throughout judging. The planned availability window ends at 17:00 on 18 October 2026, Asia/Qyzylorda, five hours after the stated noon deadline. Power/network continuity cannot be guaranteed by application code.

## Sources

- [Official Qwen fine-tuning instructions](https://github.com/QwenLM/Qwen3-VL/tree/main/qwen-vl-finetune)
- [PEFT quantized adapter training](https://huggingface.co/docs/peft/developer_guides/quantization)
- [Constrained JSON generation](https://github.com/noamgat/lm-format-enforcer/tree/817f944fcc8851917c40300a47f62d1defc3ffc3)
- [Qwen3-VL-2B publisher model card](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)
- [Qwen3-VL-4B publisher model card](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)

The supplied `ForteBank_RAG_base.zip` provides rule cards and official Adilet references. Its file hashes were checked before integration. The catalog's restrictions and `review_only` classifications are preserved. Archive validation and official-source links do not constitute a fresh legal-completeness review.
