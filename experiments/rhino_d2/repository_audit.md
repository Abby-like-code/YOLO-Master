# D2 Repository Audit

This audit is source-code based. It does not rely on README claims or external fork results.

## Repository Lock

- Source: https://github.com/Abby-like-code/YOLO-Master
- Source branch: `main`
- Source commit: `6a33fd52ce9900dfd54985a3578c33f7854e9af4`
- Source tree: `d8afbe8b3fa877259a078484e912a91d09c12ef6`
- Local checkout note: source was obtained from the GitHub branch zip because this host's `git` lacks `git-remote-https`.
- Local git branch/HEAD/status: `NOT_A_GIT_CHECKOUT`

## Environment Snapshot

See `env/environment.txt` for the full recorded environment.

- Python: `3.14.2`
- PyTorch: `2.13.0+cpu`
- CUDA runtime: `None`
- CUDA available: `False`
- GPU: `none`
- Ultralytics: `8.4.101`

## Capability Table

| Capability | Current Repo | Code Path | Used by Training Path | Admission Needed |
|------------|--------------|-----------|-----------------------|------------------|
| `FoundationDistillationModel` | Yes | `ultralytics/nn/foundation_distill_model.py` | Yes, wrapper around student model | Reused |
| `foundation_enabled` | Yes | `ultralytics/cfg/default.yaml`, `ultralytics/cfg/__init__.py` | Yes, gates trainer wrapper construction | Reused in config |
| `foundation_teacher` | Yes | `ultralytics/cfg/default.yaml`, `ultralytics/cfg/__init__.py` | Yes, selects `dinov3`, `siglip2`, or `multi` | Reused as `dinov3` |
| Teacher implementation | Yes | `ultralytics/nn/foundation/teachers/` | Yes, built by `build_foundation_distillation_wrapper` | Reused via injected DINOv3 adapter |
| DINOv2 | Not found | No matching implementation found in source audit | No | Not needed |
| DINOv3 | Yes | `ultralytics/nn/foundation/teachers/dinov3.py` | Yes | Reused |
| SigLIP / SigLIP2 | SigLIP2 yes, separate SigLIP not found | `ultralytics/nn/foundation/teachers/siglip2.py` | Yes for F12/F13/F14 | Not used in admission |
| `StudentFeatureTap` / backbone hook | Yes | `ultralytics/nn/foundation/taps.py` | Yes, wrapper creates taps for target levels | Reused for P4 |
| Projector | Yes | `ultralytics/nn/foundation/projectors.py` | Yes, wrapper-owned trainable module | Reused |
| Feature alignment | Yes | `P4AlignmentProjector.forward` | Yes, teacher spatial resize to student grid and channel projection | Reused |
| Cosine KD | Yes | `ultralytics/nn/foundation/losses.py` | Yes through wrapper dispatch | Audited, not selected for smoke |
| L2 KD | Yes | `FoundationDistillationModel._kd_components_with_weights` | Yes through wrapper dispatch | Selected for stable fixed-batch smoke |
| Relational KD | Yes | `ultralytics/nn/foundation/losses.py` | Yes through wrapper dispatch | Audited, not selected for smoke |
| Hybrid KD | Yes | `ultralytics/nn/foundation/losses.py` and wrapper dispatch | Yes | Audited, not selected for smoke |
| Foundation loss | Yes | `FoundationDistillationModel.loss` | Yes, appended to task loss item | Reused at feature-loss level |
| Trainer integration | Yes | `ultralytics/engine/trainer.py` | Yes, wraps model when foundation is active | Audited |
| Loss logging | Yes | `BaseTrainer._collect_foundation_metrics` and `_mean_foundation_metrics` | Yes, adds `train/foundation_*` metrics | Audited |
| Teacher cache | Config only | `foundation_cache_teacher_features` in `default.yaml` says future phase | No current active path found | Not used |
| Relevant tests | Yes | `tests/test_foundation_*.py` | Unit/contract coverage exists | Audited |

## Training Call Chain

The real source path is:

1. Config fields live in `ultralytics/cfg/default.yaml`.
2. `ultralytics/cfg/__init__.py` validates teacher family, backend, target levels, loss names, dtype, and incompatible legacy `distill_model`.
3. `BaseTrainer._setup_train` computes `foundation_active`.
4. If active, `BaseTrainer._setup_train` calls `build_foundation_distillation_wrapper(self.model, self.args, device=self.device)`.
5. `build_foundation_distillation_wrapper` constructs `DINOv3Teacher`, `SigLIP2Teacher`, or `MultiFoundationTeacher`, unless an injected `teacher_manager` is supplied.
6. `FoundationDistillationModel.__init__` freezes the teacher, stores it outside the registered module tree, creates `StudentFeatureTap` for `foundation_target_levels`, and builds `P4AlignmentProjector`.
7. During training, `BaseTrainer` calls `loss, self.loss_items = self.model(batch)`.
8. `FoundationDistillationModel.forward` delegates dict training batches to `FoundationDistillationModel.loss`.
9. `loss` runs the student forward, captures student P-level features, runs `teacher_manager.encode(batch["img"])` under inference mode, aligns features, dispatches KD loss, computes native task loss, and appends foundation loss to the returned loss tensor.
10. `BaseTrainer` calls `_collect_foundation_metrics`, then backpropagates `loss.sum()`.
11. Epoch-end metrics merge `train/foundation_*` into `results.csv`.

## Admission Conclusion From Audit

Current HEAD already contains a Foundation Distillation framework. The admission smoke therefore reuses existing teacher protocol, DINOv3 adapter, student tap, projector, and KD loss code. No duplicate KD framework, detection-head edits, matcher edits, MoE-router edits, multi-stage logic, or multi-teacher logic were added.

## Audit Commands

Key commands used during audit:

```powershell
rg -n "FoundationDistillationModel|foundation_enabled|foundation_teacher|DINOv3|SigLIP|StudentFeatureTap|projector|relational|hybrid|teacher cache|distill|KD" .
rg --files -g '*foundation*' -g '*distill*' -g '*teacher*' -g '*tap*' -g '*projector*' -g '*loss*' ultralytics tests docs reports
rg -n "foundation_metric|foundation_metrics|foundation_loss|loss_names|set_foundation_progress|csv" ultralytics\engine\trainer.py
rg -n "FOUNDATION_LOSSES|FOUNDATION_TEACHERS|foundation_enabled" ultralytics\cfg\__init__.py ultralytics\cfg\default.yaml
```
