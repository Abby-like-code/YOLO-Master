# D2 Foundation Distillation Admission Check

## Status

Admission smoke: PASS

## Repository

source_url: https://github.com/Abby-like-code/YOLO-Master
source_branch: main
source_commit: 6a33fd52ce9900dfd54985a3578c33f7854e9af4
source_tree: d8afbe8b3fa877259a078484e912a91d09c12ef6
local_git_branch: NOT_A_GIT_CHECKOUT
local_git_commit: NOT_A_GIT_CHECKOUT
local_dirty: false
delivery_branch: rhino-d2-admission

Local note: this host's `git` lacks `git-remote-https`, so the source was obtained from the GitHub `main` branch zip and locked to the commit above.

## Environment

Python: 3.14.2
PyTorch: 2.13.0+cpu
CUDA: None, CUDA available false
GPU: none
Ultralytics: 8.4.101

Full environment: `env/environment.txt`

## Minimal Design

Teacher: `DINOv3Teacher` adapter with injected offline DINOv3-shaped backbone
Teacher layer: `dense.p4`
Student: `yolo26-master-n`
Student stage: `p4`
Raw teacher shape: `[2, 24, 4, 4]`
Raw student shape: `[2, 128, 4, 4]`
Aligned shape: `[2, 32, 4, 4]`
Alignment: native spatial alignment `(4,4)->(4,4)` plus `P4AlignmentProjector` channels student `128->32`, teacher `24->32`
KD loss: L2 KD through existing `FoundationDistillationModel` dispatch

## Reproduction

Executed from repository root:

```powershell
python -u experiments\D2\scripts\d2_admission_smoke.py --config experiments\D2\configs\d2_admission_smoke.yaml 2>&1 | Tee-Object -FilePath experiments\D2\results\d2_admission_smoke.log
```

## Evidence

- teacher feature extraction: `results/d2_admission_smoke.json`, `teacher.raw_shape`
- student feature extraction: `results/d2_admission_smoke.json`, `student.raw_shape`
- feature alignment: `results/d2_admission_smoke.json`, `alignment`
- KD loss: `results/d2_admission_smoke.json`, `loss_history`
- gradient: `student_grad_norm` and `projector_grad_norm` in `loss_history`
- teacher frozen: `checks.teacher_frozen` and `teacher.requires_grad=false`
- teacher excluded from optimizer: `checks.teacher_excluded_from_optimizer`
- fixed-batch KD descent: `checks.kd_loss_decreased`

Loss history:

| step | raw KD loss | weighted KD loss | student grad norm | projector grad norm |
|------|-------------|------------------|-------------------|---------------------|
| 0 | 1.21445036 | 2.42890072 | 391.27840651 | 1.51780915 |
| 1 | 1.14722621 | 2.29445243 | 205.75985219 | 1.83511061 |
| 2 | 1.21575141 | 2.43150282 | 96.05488601 | 1.52994081 |
| 3 | 1.23770690 | 2.47541380 | 119.90189497 | 1.78850205 |
| 4 | 1.17367435 | 2.34734869 | 42.37610311 | 1.67029652 |
| 5 | 1.10754061 | 2.21508121 | 27.12340692 | 1.71947507 |
| 6 | 0.98864114 | 1.97728229 | 22.61565647 | 1.53475925 |
| 7 | 0.94725055 | 1.89450109 | 29.73762798 | 1.82515102 |
| 8 | 0.82060468 | 1.64120936 | 14.02194538 | 1.52956929 |
| 9 | 0.75966221 | 1.51932442 | 8.33818865 | 1.30490437 |

## No-Confound Plan

Future Baseline OFF vs KD ON experiments must keep all non-foundation fields identical: `model`, initialization, `data`, split, `seed`, `epochs`, `imgsz`, `batch`, `workers`, optimizer, learning rate, scheduler, weight decay, augmentation, AMP, deterministic settings, validation settings, and training budget.

Only existing foundation config fields may differ, such as `foundation_enabled`, `foundation_teacher`, `foundation_model`, `foundation_target_levels`, projector/alignment fields, `foundation_loss`, and `foundation_loss_weight`. See `admission_design.md` for the complete allowlist.

## Current Scope

This is only an admission smoke.

No mAP claim.
No P0 completion claim.
No P1 experiment has been executed.

## Next Step

The next stage is D2 P0. Stop here for admission.
