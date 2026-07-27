#!/usr/bin/env python3
"""Train the Ours (slim) family or a stock YOLOv11 baseline on VisDrone2019, config-driven.

Config layering: configs/base.yaml (defaults) <- configs/exp<NN>.yaml (1 file cho mỗi
dòng thí nghiệm, trỏ thẳng bằng --config; hoặc configs/<idea>.yaml auto-resolved từ
--idea) <- CLI flags (override cuối cùng, chỉ áp khi thực sự được truyền).

Idea nào cũng đăng ký trong models/ (xem models/__init__.py MODEL_REGISTRY) —
--idea nhận choices trực tiếp từ đó, thêm idea mới không cần sửa file này.

Mỗi config ứng với ĐÚNG một dòng trong results/SOTA-SLIM.xlsx:

    Bảng 2 — đổi block (2x2 factorial: upsampler x block)
      configs/exp01.yaml   DySample + C3k2Ghost (SLIM)      0.557M / 6.16G   [đã chạy]
      configs/exp02.yaml   DySample + C3k2Rep               0.900M / 9.21G
      configs/exp03.yaml   bilinear + C3k2Ghost             0.550M / 6.18G
      configs/exp04.yaml   bilinear + C3k2Rep = Ours-N      0.893M / 9.23G   [đã chạy]

    Bảng 3 — scale
      configs/exp06.yaml   Ours-S                           2.250M / 22.31G
      configs/exp07.yaml   Ours-M                (optional) 3.994M / 39.37G
      configs/exp08.yaml   YOLOv11n from-scratch            2.584M / 6.32G   [đã chạy]
      configs/exp09.yaml   YOLOv11s from-scratch            9.417M / 21.32G  [đã chạy]
      configs/exp10.yaml   YOLOv11m from-scratch (optional) 20.04M / 67.69G

    Bảng 4 — gỡ từng thành phần khỏi Ours-N
      configs/exp12.yaml   - LSCDetect (stock Detect)       1.316M / 15.68G
      configs/exp13.yaml   - nhánh P2 (head P3+P4)          0.774M / 5.47G
      configs/exp14.yaml   - NWD (CIoU thuần, chỉ đổi loss) 0.893M / 9.23G
      configs/exp15.yaml   - attention (EMA+C2PSA+ECA)      0.827M / 8.77G
      configs/exp16.yaml   - SPDConv (Conv stride-2)        1.063M / 10.26G
      configs/exp17.yaml   head GroupNorm -> BatchNorm      0.893M / 9.23G

(params/GFLOPs ở trên là bản CHƯA fuse, đo @640 — xem tests/test_modules.py, nơi
regression-guard đúng những con số này.)

Examples:
    # chạy 1 dòng trong bảng
    python train.py --config configs/exp02.yaml

    # override vài hyperparam từ CLI, đè lên config
    python train.py --config configs/exp02.yaml --imgsz 640 --batch 16

    # log mAP tập test mỗi 10 epoch (monitoring, không dùng để chọn best.pt)
    python train.py --config configs/exp04.yaml --test-every 10
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

# Anaconda ships its own libiomp5md.dll which clashes with torch's copy on Windows
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# ultralytics trains with torch.use_deterministic_algorithms(True, warn_only=True);
# DySample's grid_sample and EMA's adaptive_avg_pool2d have no deterministic CUDA
# backward, so torch warns every epoch. Harmless (warn_only falls back to the
# non-deterministic kernel) — silence it instead of dropping deterministic=True.
warnings.filterwarnings(
    "ignore", message=".*does not have a deterministic implementation.*"
)

# Windows' default console codepage (e.g. cp1258) can't encode the Vietnamese
# diacritics in --help text / log messages below; force UTF-8 so `--help` and
# prints don't crash instead of silently mojibake-ing.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent


def load_config(base_cfg: str, exp_cfg: str):
    """Merge configs/base.yaml + configs/<idea>.yaml. Priority: exp_cfg > base_cfg."""
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(base_cfg)
    cfg = OmegaConf.merge(cfg, OmegaConf.load(exp_cfg))
    return cfg


def main():
    from models import MODEL_REGISTRY

    ap = argparse.ArgumentParser()
    ap.add_argument("--idea", default="baseline", choices=sorted(MODEL_REGISTRY),
                     help="chọn configs/<idea>.yaml + models/<idea>.py (bỏ qua nếu dùng --config)")
    ap.add_argument("--config", default=None,
                     help="trỏ thẳng 1 experiment yaml, bỏ qua auto-resolve theo --idea")
    ap.add_argument("--base-config", default=str(ROOT / "configs" / "base.yaml"))

    # Mọi flag dưới đây override giá trị tương ứng trong config CHỈ KHI được truyền
    # (default=None => "không truyền" khác với "truyền giá trị falsy")
    ap.add_argument("--model", default=None, dest="model_cfg",
                     help="model .yaml (train from scratch) hoặc .pt (finetune) — xem "
                          "danh sách option ở docstring đầu file")
    ap.add_argument("--data", default=None)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None,
                     help="P2 head @960 is VRAM-hungry; 8 fits ~12GB, raise if you can")
    ap.add_argument("--device", default=None, help="e.g. 0 or 0,1 or cpu")
    ap.add_argument("--name", default=None, help="run name under runs/detect/")
    ap.add_argument("--weights", default=None, help="optional .pt to warm-start a .yaml model")
    ap.add_argument("--multi-scale", dest="multi_scale", action="store_true", default=None,
                     help="multi-scale training (more VRAM)")
    ap.add_argument("--close-mosaic", type=int, default=None)
    ap.add_argument("--mixup", type=float, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--resume", action="store_true", default=None)
    ap.add_argument("--seed", type=int, default=None, help="random seed (reproducibility)")
    ap.add_argument("--test-every", type=int, default=None, dest="test_every",
                     help="also val on the test-dev split every N epochs (monitoring only; "
                          "0 disables). Adds one extra full test-set pass per trigger, on top "
                          "of the normal per-epoch val — costs time, use a coarse N (e.g. 10-20).")
    # NWD blend loss (see models/lwso/losses.py) — --no-nwd always forces it off regardless of config
    ap.add_argument("--no-nwd", action="store_true", help="use stock CIoU loss only")
    ap.add_argument("--nwd-ratio", type=float, default=None, help="weight of CIoU term in the blend")
    ap.add_argument("--nwd-constant", type=float, default=None, help="NWD normalizing constant C")
    args = ap.parse_args()

    exp_cfg = args.config or str(ROOT / "configs" / f"{args.idea}.yaml")
    # Experiment configs are named per table row (configs/exp<NN>.yaml), so the old
    # configs/<idea>.yaml auto-resolve no longer finds anything -- say so plainly instead
    # of dying on a bare FileNotFoundError.
    if not Path(exp_cfg).is_file():
        available = ", ".join(sorted(p.name for p in (ROOT / "configs").glob("exp*.yaml")))
        raise SystemExit(
            f"config không tồn tại: {exp_cfg}\n"
            f"Mỗi thí nghiệm là 1 file riêng, chạy bằng --config. Có sẵn: {available}"
        )
    cfg = load_config(args.base_config, exp_cfg)
    # Recorded in logs/<name>.log's header so a log always says which config produced it.
    cfg["config_file"] = str(Path(exp_cfg).relative_to(ROOT)) if Path(exp_cfg).is_relative_to(ROOT) else exp_cfg

    overrides = {
        "model_cfg": args.model_cfg, "data": args.data, "imgsz": args.imgsz,
        "epochs": args.epochs, "batch": args.batch, "device": args.device,
        "name": args.name, "weights": args.weights,
        "multi_scale": args.multi_scale,
        "close_mosaic": args.close_mosaic, "mixup": args.mixup, "patience": args.patience,
        "resume": args.resume, "seed": args.seed, "test_every": args.test_every,
        "nwd_ratio": args.nwd_ratio, "nwd_constant": args.nwd_constant,
    }
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v
    if args.no_nwd:
        cfg.use_nwd = False

    print(f"[lwso] idea={cfg.idea}  config={exp_cfg}")

    from models import build_model

    model = build_model(cfg.idea, cfg)
    model.train(
        data=str(cfg.data),
        imgsz=int(cfg.imgsz),
        epochs=int(cfg.epochs),
        batch=int(cfg.batch),
        device=cfg.device,
        name=str(cfg.name),
        close_mosaic=int(cfg.close_mosaic),
        mixup=float(cfg.mixup),
        multi_scale=bool(cfg.multi_scale),
        patience=int(cfg.patience),
        resume=bool(cfg.resume),
        seed=int(cfg.seed),
        cos_lr=True,
        plots=True,
    )


if __name__ == "__main__":
    main()
