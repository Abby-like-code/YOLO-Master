# D2 Admission Smoke Design

This document covers only the D2 Foundation Distillation admission check.

## Scope

The smoke validates the existing feature-KD chain on one fixed batch:

- one student: `yolo26-master-n`
- one teacher adapter: `DINOv3Teacher` with an injected offline backbone
- one student stage: `p4`
- one teacher dense feature: `dense.p4`
- one alignment module: `P4AlignmentProjector`
- one KD loss: `l2`
- one fixed batch: two deterministic `64x64` images
- ten optimizer steps

No detection training, COCO/coco128 epoch, mAP, P0, P1, multi-seed, multi-stage, multi-teacher, relational/hybrid ablation, or teacher-cache experiment is executed.

## Why An Injected DINOv3 Adapter

The repository already supports a `teacher_manager` injection point. The admission smoke uses the real `DINOv3Teacher` adapter while injecting a tiny local backbone that emits DINOv3-shaped dense maps. This proves the current adapter/protocol/alignment/loss/backward path without making internet downloads, Hugging Face cache state, or pretrained DINOv3 weights part of admission.

The following remain explicitly not verified:

- real pretrained DINOv3 weights
- Hugging Face model download/cache behavior
- full detection-loss integration on a real detection batch
- mAP or accuracy improvement

## Actual Smoke Path

1. Build `YOLO(ROOT / "ultralytics/cfg/models/26/yolo26-master-n.yaml").model`.
2. Build `DINOv3Teacher(model=OfflineDINOv3Backbone(...), device="cpu", dtype="fp32")`.
3. Wrap student and teacher with `FoundationDistillationModel`.
4. Run the fixed batch through `teacher_manager.encode`.
5. Run the same fixed batch through the student and capture P4 with `StudentFeatureTap`.
6. Align with `P4AlignmentProjector`.
7. Compute wrapper-dispatched L2 KD.
8. Backpropagate weighted KD loss through student and projector only.
9. Verify teacher gradients remain absent/zero and teacher parameters are not in the optimizer.

## Selected Shapes

Observed shapes from `results/d2_admission_smoke.json`:

| Tensor | Shape |
|--------|-------|
| input | `[2, 3, 64, 64]` |
| teacher raw `dense.p4` | `[2, 24, 4, 4]` |
| student raw `p4` | `[2, 128, 4, 4]` |
| teacher aligned | `[2, 32, 4, 4]` |
| student aligned | `[2, 32, 4, 4]` |

Spatial alignment was native, with no resize: teacher `(4, 4)` to student `(4, 4)`. Channel alignment used the existing `P4AlignmentProjector`: student `128 -> 32`, teacher `24 -> 32`.

## Pass Criteria

The smoke passes only if all of these are true in JSON:

- teacher dense feature extracted
- student P4 feature extracted
- aligned shapes equal
- KD loss finite
- KD loss non-zero
- teacher frozen
- teacher excluded from optimizer
- student gradient non-zero
- projector gradient non-zero
- final KD loss below initial KD loss

## No-Confound Plan For Later P0/P1

Future Baseline OFF vs KD ON experiments should be paired configs. All non-foundation fields must remain identical.

| Field group | Baseline OFF | KD ON | Constraint |
|-------------|--------------|-------|------------|
| student model | same `model` | same `model` | identical |
| initialization | same `pretrained` / checkpoint | same `pretrained` / checkpoint | identical |
| dataset | same `data` | same `data` | identical |
| split | same `split`, train/val files | same `split`, train/val files | identical |
| seed | same `seed` | same `seed` | identical |
| epochs | same `epochs` | same `epochs` | identical |
| image size | same `imgsz` | same `imgsz` | identical |
| batch | same `batch` | same `batch` | identical |
| workers | same `workers` | same `workers` | identical |
| optimizer | same `optimizer` | same `optimizer` | identical |
| learning rate | same `lr0`, `lrf` | same `lr0`, `lrf` | identical |
| scheduler | same `cos_lr`, warmup fields | same `cos_lr`, warmup fields | identical |
| weight decay | same `weight_decay` | same `weight_decay` | identical |
| augmentation | same HSV/mosaic/copy-paste/mixup/etc. fields | same fields | identical |
| AMP | same `amp` | same `amp` | identical |
| deterministic | same `deterministic` | same `deterministic` | identical |
| validation | same `val`, `conf`, `iou`, `max_det`, `save_json` | same fields | identical |
| training budget | same hardware/time/fraction | same hardware/time/fraction | identical |

Only these current repository fields may differ:

- `foundation_enabled`
- `foundation_teacher`
- `foundation_backend`
- `foundation_model`
- `foundation_weights`
- `foundation_dinov3_model`
- `foundation_siglip2_model`
- `foundation_dinov3_weights`
- `foundation_siglip2_weights`
- `foundation_target_levels`
- `foundation_multiscale`
- `foundation_align_dim`
- `foundation_loss`
- `foundation_loss_weight`
- `foundation_cosine_weight`
- `foundation_relation_weight`
- `foundation_relation_mode`
- `foundation_relation_samples`
- `foundation_foreground_weighting`
- `foundation_foreground_weight`
- `foundation_boundary_weight`
- `foundation_background_weight`
- `foundation_teacher_dtype`
- `foundation_teacher_device`
- `foundation_router_distill`
- `foundation_router_loss_weight`
- `foundation_router_temperature`
- `foundation_router_teachers`
- `foundation_router_native_state`
- `foundation_semantic_distill`
- `foundation_semantic_loss_weight`
- `foundation_semantic_text_weight`
- `foundation_semantic_image_weight`
- `foundation_semantic_temperature`
- `foundation_semantic_prompts`
- `foundation_semantic_prompt_template`
- `foundation_weight_schedule`
- `foundation_gate_cosine`
- `foundation_gate_cosine_low`
- `foundation_gate_width`
- `foundation_warmup_floor`
- `foundation_decay_start`
- `foundation_gate_ema`

`foundation_cache_teacher_features` exists in config but is marked as a future phase and should stay off unless cache behavior itself becomes the experiment.

## Result

Admission smoke status: `passed`.

This is only an admission smoke. It makes no mAP claim, no P0 completion claim, and no P1 experiment claim.
