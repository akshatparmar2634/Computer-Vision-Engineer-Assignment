# Custom VLM Design for PCB Quality Inspection

## Overview

- Goal: Offline VLM that answers defect-related natural language questions with structured boxes and confidence, <2s latency on edge GPU/CPU.
- Data: 50k PCB images with defect boxes (no QA). Need grounding-first design to avoid hallucination and support precise localization.

## (A) Model Selection

- Choice: Start from Qwen2-VL-7B base (or comparable 7B-class open model) plus custom grounding head.
  - Rationale: strong vision-language grounding, permissive license, efficient int4/int8 support, good instruction-following, robust multilingual tokenizer.
  - Size vs speed: 7B fits edge GPUs with int4 and TensorRT; larger (13B+) risks >2s. For CPU-only, consider 3B distilled variant.
  - Fine-tuning flexibility: supports LoRA/QLoRA, mixed-precision, custom adapter layers on vision and cross-attn blocks.
  - Licensing: Qwen2-VL license permits offline commercial use; verify version and redistribution terms.
- Architectural needs for localization:
  - Add detection-style grounding tokens and a lightweight box head over vision encoder features.
  - Use token-level pointer regression (e.g., GLIGEN-style box tokens) to emit coordinates; constrain outputs to structured JSON.
  - Multi-scale features (FPN) from vision encoder to capture small defects.

## (B) Architecture Design

- Vision encoder: SigLIP or ViT-L/14 backbone with FPN neck; keep patch stride small for fine features (defect pixels). Frozen early, fine-tune neck.
- Detection head: YOLOv8n-style or DINO-style query decoder producing K region proposals. Expose region embeddings to language model.
- Fusion: Cross-attention adapters that take top-K region embeddings plus question tokens; add relative positional encoding for box coordinates.
- Language decoder: Qwen2-VL LLM with constrained decoding to structured schema (JSON or YAML). Add special tokens: [BOX], [CLS], [SCORE].
- Output schema: {"answer": string, "regions": [{"label": str, "bbox": [x1,y1,x2,y2], "confidence": float}]}
- Retrieval prior: optional defect-class prior table to bias decoder toward known classes.

## (C) Optimization for <2s Offline

- Quantization: INT8 vision encoder; INT4/FP8 language decoder via TensorRT/ONNX Runtime; smooth-quant for stable activation ranges.
- Pruning: Structured pruning on cross-attn and FFN blocks (10-20%) followed by brief recovery finetune.
- Distillation: Distill 7B to 3B-4B student with KD on logits and box regression (L1/GIoU) to hit CPU targets.
- Caching: Cache vision features; for multi-question on same image, reuse region embeddings.
- Efficient decoding: Constrained beam search with small beam (1-2), top-p 0.8, max tokens ~128; dynamic kv-cache; fused kernels.
- Batch: Micro-batching of 2-4 images if hardware allows; otherwise single-stream.

## (D) Hallucination Mitigation

- Training signals:
  - Grounding loss: L1 + GIoU on predicted boxes vs GT; focal loss on class logits.
  - Answer grounding consistency: force every mentioned defect to align to a predicted box (coverage loss).
  - No-defect supervision: include clean boards; penalize false positives with calibrated focal negatives.
  - Contrastive negs: hard negative prompts where defect is absent; optimize with binary cross-entropy on presence tokens.
- Architectural:
  - Structured decoding with schema-constrained grammar; disallow free-form generation when answering localization queries.
  - Region-only attention mode: language tokens attend only to selected region embeddings (not global CLS) to reduce free-form hallucination.
- Decoding controls: temperature 0.2-0.4; refuse/"not detected" token encouraged via special token with calibrated prior.

## (E) Training Plan

1) Detection pretraining (supervised):
   - Use 50k boxes; train vision plus FPN plus detection head with YOLO/DINO losses to strong mAP.
2) Synthetic QA generation:
   - Template and LLM-generated questions per image: presence, count, location, class, severity. Derive answers directly from annotations to avoid noise.
   - Generate negative QAs (asking for absent defects) for hallucination control.
3) Instruction tuning (LoRA/QLoRA on LLM plus cross-attn):
   - Mix detection tokens with text; losses: LM cross-entropy on answers, box regression plus coverage.
4) Preference/RL stage (optional):
   - Use rule-based reward: penalize hallucinated classes/boxes; reward precise IoU and count accuracy; apply DPO/ORPO or RLHF light.
5) Data augmentation:
   - Geometric: flip, rotate small angles, scale jitter, mosaic/cutout while preserving boxes.
   - Photometric: brightness/contrast jitter, Gaussian noise to mimic lighting.
   - Defect copy-paste to balance rare classes.
6) Curriculum:
   - Start with single-defect questions; progress to multi-defect counting and compound queries.
7) Checkpoints:
   - Save best on mAP and hallucination rate; export ONNX/TensorRT engines.

## (F) Validation

- Counting: MAE and RMSE of defect counts per class; F1 on count correctness.
- Localization: mAP@0.5 and mAP@0.5:0.95; mean IoU on answered boxes; recall@K for mentioned defects.
- QA correctness: Exact-match/ROUGE-L on textual answers constrained to schema.
- Hallucination: False positive rate on negative QAs; Hallucination@k = (# answers with non-existent defects)/(# negative queries); calibration curves for confidence.
- Latency: P95 end-to-end response time on target hardware with warm cache; measure with and without vision-cache reuse.
- Ablations: with/without schema constraints; different quant levels; student vs teacher.

## Inference Pipeline

- Step 1: Encode image -> vision features -> detection head -> top-K regions plus boxes.
- Step 2: Fuse question tokens with region embeddings via cross-attn adapters.
- Step 3: Constrained decode to schema; post-process to numeric bbox array; filter by confidence threshold tuned for low FP.

## Deployment Notes

- Package as TensorRT/ONNX runtime; include grammar-constrained decoder.
- Provide offline validator to reject untrusted prompts (only allow defect-related intents).
- Ship a calibration set per hardware for quantization and threshold tuning.
