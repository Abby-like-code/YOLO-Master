"""D2 Foundation distillation admission smoke.

This script intentionally validates only the feature-KD admission path. It does
not run detection training, COCO, mAP, multi-seed, or P0/P1 experiments.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".codex_yolo_config"))
(Path(os.environ["YOLO_CONFIG_DIR"])).mkdir(parents=True, exist_ok=True)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
import yaml
from torch import nn

import ultralytics
from ultralytics import YOLO
from ultralytics.nn.foundation import DINOv3Teacher
from ultralytics.nn.foundation_distill_model import FoundationDistillationModel


def _run(command: list[str]) -> str:
    try:
        completed = subprocess.run(command, cwd=ROOT, check=True, text=True, capture_output=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        return f"UNAVAILABLE: {exc}"
    return completed.stdout.strip()


def _run_project_git(args: list[str]) -> str:
    if not (ROOT / ".git").exists():
        return "NOT_A_GIT_CHECKOUT"
    return _run(["git", *args])


def _shape(tensor: torch.Tensor | None) -> list[int] | None:
    return list(tensor.shape) if isinstance(tensor, torch.Tensor) else None


def _grad_norm(parameters) -> float:
    total = 0.0
    for parameter in parameters:
        grad = parameter.grad
        if grad is None:
            continue
        total += float(grad.detach().float().pow(2).sum().item())
    return total**0.5


def _requires_grad(parameters) -> bool:
    return any(parameter.requires_grad for parameter in parameters)


def _grad_all_none_or_zero(parameters) -> bool:
    for parameter in parameters:
        grad = parameter.grad
        if grad is not None and float(grad.detach().abs().max().item()) != 0.0:
            return False
    return True


class OfflineDINOv3Backbone(nn.Module):
    """Small local backbone shaped like a DINOv3 ViT output for adapter smoke."""

    def __init__(self, hidden_size: int, patch_size: int) -> None:
        super().__init__()
        self.config = SimpleNamespace(patch_size=patch_size, hidden_size=hidden_size, num_register_tokens=0)
        self.proj = nn.Conv2d(3, hidden_size, kernel_size=1, bias=False)
        with torch.no_grad():
            values = torch.linspace(-0.4, 0.4, steps=hidden_size * 3, dtype=torch.float32)
            self.proj.weight.copy_(values.reshape(hidden_size, 3, 1, 1))

    def forward(self, pixel_values: torch.Tensor):
        patch = int(self.config.patch_size)
        pooled = F.avg_pool2d(pixel_values, kernel_size=patch, stride=patch)
        feature = self.proj(pooled)
        return SimpleNamespace(feature_maps=(feature,), pooler_output=feature.mean(dim=(2, 3)))


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _fixed_batch(batch_size: int, imgsz: int) -> dict[str, torch.Tensor]:
    total = batch_size * 3 * imgsz * imgsz
    image = torch.linspace(0.0, 1.0, steps=total, dtype=torch.float32).reshape(batch_size, 3, imgsz, imgsz)
    return {"img": image}


def _environment() -> dict[str, Any]:
    return {
        "python": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "ultralytics": ultralytics.__version__,
        "ultralytics_file": str(Path(ultralytics.__file__).resolve()),
    }


def _write_environment(path: Path, env: dict[str, Any], repo: dict[str, Any]) -> None:
    lines = ["# D2 Admission Environment", ""]
    lines.extend(f"{key}: {value}" for key, value in repo.items())
    lines.append("")
    lines.extend(f"{key}: {value}" for key, value in env.items())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--json-out", type=Path, default=ROOT / "experiments/D2/results/d2_admission_smoke.json")
    parser.add_argument("--env-out", type=Path, default=ROOT / "experiments/D2/env/environment.txt")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    runtime = cfg["runtime"]
    foundation = cfg["foundation"]
    torch.manual_seed(int(runtime["seed"]))
    random.seed(int(runtime["seed"]))
    torch.set_num_threads(1)

    repo = {
        "remote": _run_project_git(["remote", "-v"]),
        "branch": _run_project_git(["branch", "--show-current"]),
        "head": _run_project_git(["rev-parse", "HEAD"]),
        "status_short": _run_project_git(["status", "--short"]),
        "source_url": cfg["repository"]["source_url"],
        "source_branch": cfg["repository"]["source_branch"],
        "source_commit": cfg["repository"]["source_commit"],
        "source_tree": cfg["repository"]["source_tree"],
        "local_checkout_note": "source obtained from GitHub main zip because local git lacks remote-https helper",
    }
    env = _environment()
    args.env_out.parent.mkdir(parents=True, exist_ok=True)
    _write_environment(args.env_out, env, repo)

    print("D2 Foundation Distillation Admission Check")
    print(f"claim: {cfg['claim']}")
    print("repository:")
    for key, value in repo.items():
        print(f"  {key}: {value}")
    print("environment:")
    for key, value in env.items():
        print(f"  {key}: {value}")

    teacher_cfg = cfg["teacher"]
    teacher_backbone = OfflineDINOv3Backbone(
        hidden_size=int(teacher_cfg["hidden_size"]),
        patch_size=int(teacher_cfg["patch_size"]),
    )
    teacher = DINOv3Teacher(
        model_id=str(teacher_cfg["model_id"]),
        model=teacher_backbone,
        dtype="fp32",
        device=str(runtime["device"]),
    )

    student_cfg = cfg["student"]
    student = YOLO(str(ROOT / student_cfg["config"])).model
    student.train()

    wrapper = FoundationDistillationModel(student, teacher, SimpleNamespace(**foundation, imgsz=runtime["imgsz"]))
    wrapper.train()

    batch = _fixed_batch(int(runtime["batch_size"]), int(runtime["imgsz"]))
    batch = {key: value.to(runtime["device"]) for key, value in batch.items()}
    teacher_output = wrapper.teacher_manager.encode(batch["img"])
    teacher_raw = teacher_output.dense["p4"]

    for tap in wrapper.taps.values():
        tap.clear()
    student_pred = wrapper.student_model(batch["img"])
    student_raw = wrapper.tap.feature
    student_aligned, teacher_aligned = wrapper.projector(student_raw, teacher_raw)
    alignment = wrapper.projector.alignment

    print("teacher:")
    print(f"  model: {teacher.__class__.__name__}")
    print(f"  layer: {teacher_cfg['layer']}")
    print(f"  raw_feature_shape: {_shape(teacher_raw)}")
    print(f"  requires_grad: {_requires_grad(teacher.parameters())}")
    print(f"  training: {teacher.training}")
    print("student:")
    print(f"  model: {student_cfg['name']}")
    print(f"  stage: {student_cfg['stage']}")
    print(f"  raw_feature_shape: {_shape(student_raw)}")
    print("alignment:")
    print(f"  student_aligned_shape: {_shape(student_aligned)}")
    print(f"  teacher_aligned_shape: {_shape(teacher_aligned)}")
    print(
        "  spatial: teacher "
        f"{alignment['teacher_size']} -> {alignment['target_size']} "
        f"mode={'bilinear' if alignment['teacher_resized'] else 'native'}"
    )
    print(
        "  channel: student "
        f"{wrapper.projector.student_channels} -> {wrapper.projector.align_dim}; "
        f"teacher {wrapper.projector.teacher_channels} -> {wrapper.projector.align_dim}"
    )
    print(f"  projector_trainable: {_requires_grad(wrapper.projector.student_proj.parameters())}")
    print(f"  teacher_projection_frozen: {wrapper.projector.teacher_projection_frozen}")
    print(f"  student_prediction_type: {type(student_pred).__name__}")

    optimizer = torch.optim.SGD([p for p in wrapper.parameters() if p.requires_grad], lr=float(runtime["lr"]))
    loss_history = []
    teacher_in_optimizer = any(
        id(parameter) in {id(p) for group in optimizer.param_groups for p in group["params"]}
        for parameter in teacher.parameters()
    )

    for step in range(int(runtime["steps"])):
        optimizer.zero_grad(set_to_none=True)
        for tap in wrapper.taps.values():
            tap.clear()
        wrapper.student_model(batch["img"])
        teacher_output = wrapper.teacher_manager.encode(batch["img"])
        teacher_feature = teacher_output.dense["p4"]
        student_feature = wrapper.tap.feature
        student_aligned, teacher_aligned = wrapper.projector(student_feature, teacher_feature)
        kd_loss = wrapper._kd_components(student_aligned, teacher_aligned)[0]
        weighted_kd_loss = kd_loss * wrapper.effective_loss_weight() * int(batch["img"].shape[0])
        weighted_kd_loss.backward()
        student_grad_norm = _grad_norm(wrapper.student_model.parameters())
        projector_grad_norm = _grad_norm(wrapper.projector.student_proj.parameters())
        teacher_grad_clear = _grad_all_none_or_zero(teacher.parameters())
        optimizer.step()
        record = {
            "step": step,
            "kd_loss": float(kd_loss.detach().item()),
            "weighted_kd_loss": float(weighted_kd_loss.detach().item()),
            "student_grad_norm": student_grad_norm,
            "projector_grad_norm": projector_grad_norm,
            "teacher_grad_clear": teacher_grad_clear,
        }
        loss_history.append(record)
        print(
            "step "
            f"{step}: raw_kd_loss={record['kd_loss']:.8f} "
            f"weighted_kd_loss={record['weighted_kd_loss']:.8f} "
            f"student_grad_norm={student_grad_norm:.8f} "
            f"projector_grad_norm={projector_grad_norm:.8f} "
            f"teacher_grad_clear={teacher_grad_clear}"
        )

    losses = [record["kd_loss"] for record in loss_history]
    checks = {
        "teacher_feature_extracted": isinstance(teacher_raw, torch.Tensor) and teacher_raw.ndim == 4,
        "student_feature_extracted": isinstance(student_raw, torch.Tensor) and student_raw.ndim == 4,
        "alignment_success": list(student_aligned.shape) == list(teacher_aligned.shape),
        "finite_loss": all(torch.isfinite(torch.tensor(loss)).item() for loss in losses),
        "nonzero_loss": any(abs(loss) > 0 for loss in losses),
        "teacher_frozen": (not _requires_grad(teacher.parameters())) and (not teacher.training),
        "teacher_excluded_from_optimizer": not teacher_in_optimizer,
        "student_gradient": any(record["student_grad_norm"] > 0 for record in loss_history),
        "projector_gradient": any(record["projector_grad_norm"] > 0 for record in loss_history),
        "kd_loss_decreased": losses[-1] < losses[0],
    }
    status = "passed" if all(checks.values()) else "failed"
    result = {
        "status": status,
        "claim": cfg["claim"],
        "repository": {
            "branch": repo["branch"],
            "commit": repo["head"],
            "dirty": bool(repo["status_short"] and repo["status_short"] != "NOT_A_GIT_CHECKOUT"),
            "source_url": repo["source_url"],
            "source_branch": repo["source_branch"],
            "source_commit": repo["source_commit"],
            "source_tree": repo["source_tree"],
            "local_checkout_note": repo["local_checkout_note"],
        },
        "environment": env,
        "teacher": {
            "name": teacher_cfg["name"],
            "adapter": teacher.__class__.__name__,
            "model_id": teacher.model_id,
            "layer": teacher_cfg["layer"],
            "raw_shape": _shape(teacher_raw),
            "requires_grad": _requires_grad(teacher.parameters()),
            "training": teacher.training,
            "metadata": teacher_output.metadata,
        },
        "student": {
            "name": student_cfg["name"],
            "config": student_cfg["config"],
            "stage": student_cfg["stage"],
            "tap_source_index": wrapper.tap.source_index,
            "tap_source_indices": list(wrapper.tap.source_indices),
            "raw_shape": _shape(student_raw),
        },
        "alignment": {
            "teacher_shape": _shape(teacher_aligned),
            "student_shape": _shape(student_aligned),
            "spatial_method": "bilinear_resize_teacher_to_student"
            if alignment["teacher_resized"]
            else "native_no_resize",
            "spatial_from_shape": list(alignment["teacher_size"]),
            "spatial_to_shape": list(alignment["target_size"]),
            "channel_method": "P4AlignmentProjector student 1x1 Conv+BN; frozen teacher 1x1 Conv/Identity",
            "student_channels": wrapper.projector.student_channels,
            "teacher_channels": wrapper.projector.teacher_channels,
            "align_dim": wrapper.projector.align_dim,
            "projector_trainable": _requires_grad(wrapper.projector.student_proj.parameters()),
            "teacher_projection_frozen": wrapper.projector.teacher_projection_frozen,
        },
        "checks": checks,
        "loss_history": loss_history,
        "not_verified": [
            "real pretrained DINOv3 weights",
            "COCO/coco128 detection training",
            "mAP or accuracy improvement",
            "P0/P1 experiment",
            "multi-seed statistics",
            "multi-stage or multi-teacher ablations",
            "foundation_in_total_loss via full detection loss",
        ],
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("final admission checks:")
    for key, value in checks.items():
        print(f"  {key}: {value}")
    print(f"status: {status}")
    print(f"json_out: {args.json_out}")
    print(f"env_out: {args.env_out}")
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
