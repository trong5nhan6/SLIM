#!/usr/bin/env python3
"""Evaluate a trained checkpoint on VisDrone2019 val or test-dev.

Always reports efficiency metrics (params/GFLOPs/latency/FPS) alongside mAP, matching
the block train.py's --test-every callback writes during training (models/base_model.py)
-- so a standalone eval after training gives the same numbers, not just mAP.

One run of this script prints every column a results table needs -- P, R, mAP50,
mAP50-95, params, GFLOPs, model size, FPS, epochs, imgsz -- ending with a single
comma-separated PAPER TABLE ROW to paste straight into a spreadsheet. Precision/Recall
and the train-time settings used to live only in ultralytics' own printout and in the
checkpoint respectively, which meant transcribing them by hand for every run.

    python val.py --weights runs/detect/exp04/weights/best.pt --split val
    python val.py --weights runs/detect/exp04/weights/best.pt --split test
"""

import argparse
import os
from pathlib import Path

# Anaconda ships its own libiomp5md.dll which clashes with torch's copy on Windows
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parent


def _read_train_args(weights: str) -> dict:
    """Recover the recipe a checkpoint was trained with (epochs/imgsz/batch/seed).

    Ultralytics stores the full training namespace under the checkpoint's `train_args`
    key, so the table columns that describe HOW a model was trained never have to be
    remembered -- they travel with the .pt. Returns {} for anything unreadable rather
    than failing the eval over a reporting nicety.
    """
    import torch

    try:
        ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    except Exception:
        return {}
    args = ckpt.get("train_args") or {}
    return dict(args) if hasattr(args, "keys") else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default=str(ROOT / "data" / "visdrone.yaml"))
    ap.add_argument("--imgsz", type=int, default=640,
                     help="phải khớp imgsz lúc train (mặc định 640 = imgsz của mọi dòng "
                          "trong results/SOTA-SLIM.xlsx); lệch thì GFLOPs/latency vô nghĩa")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument("--log-name", default=None, dest="log_name",
                     help="ghi ket qua vao logs/<ten>.log; mac dinh suy ra tu ten run trong "
                          "duong dan --weights (runs/detect/<ten>/weights/best.pt)")
    ap.add_argument("--no-log", action="store_true", help="chi in ra console, khong ghi file log")
    args = ap.parse_args()

    # Register the custom modules regardless of which config produced this checkpoint --
    # cheap, idempotent, harmless for a stock (idea baseline) checkpoint that uses none of
    # them. Loading a .pt only unpickles the classes, but a checkpoint that ever has to be
    # rebuilt from its YAML needs parse_model to know them too. register_slim() calls
    # register_lwso() internally for the shared SPDConvGroup/EMA/ECA/DySample/BiFPNCat set.
    from models.slim.register import register_slim

    register_slim()

    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import select_device

    from models.base_model import (
        _compute_efficiency_metrics,
        _format_efficiency_report,
        append_log,
        log_banner,
        log_path_for,
        log_row as row,
    )

    # Everything worth keeping goes through emit(): printed now, and appended to
    # logs/<run>.log at the end so one config ends up with ONE log holding the training
    # trace (written by the train callbacks) plus every eval ever run on it.
    report: list[str] = []

    def emit(lines) -> None:
        lines = [lines] if isinstance(lines, str) else lines
        report.extend(lines)
        print("\n".join(lines))

    # runs/detect/<name>/weights/best.pt -> <name>; falls back to the file stem for a
    # checkpoint sitting somewhere else (e.g. weights/best_exp04.pt).
    run_name = args.log_name
    if run_name is None:
        wp = Path(args.weights).resolve()
        run_name = wp.parents[1].name if wp.parent.name == "weights" else wp.stem

    train_args = _read_train_args(args.weights)
    emit(log_banner(f"EVAL {run_name}  split={args.split}"))
    emit([row("Weights", args.weights)])
    emit([row(label, train_args.get(key, "?")) for label, key in
          (("Train epochs", "epochs"), ("Train imgsz", "imgsz"),
           ("Train batch", "batch"), ("Seed", "seed"))])
    emit([row("Eval split", args.split), row("Eval imgsz", args.imgsz)])
    # GFLOPs and latency are both quadratic-ish in imgsz, so evaluating at a different
    # size than training silently produces numbers that belong in a different table.
    if train_args.get("imgsz") not in (None, args.imgsz):
        emit(f"\n  !! WARNING: eval imgsz {args.imgsz} != train imgsz {train_args['imgsz']} "
             f"-- efficiency numbers will not match the training recipe.")
    print()

    model = YOLO(args.weights)
    metrics = model.val(
        data=args.data, imgsz=args.imgsz, split=args.split, batch=args.batch, device=args.device
    )
    # mp/mr are the class-averaged P/R at the F1-optimal confidence threshold (ultralytics
    # prints them in the `all` row); unlike mAP they depend on that threshold.
    p, r = metrics.box.mp * 100, metrics.box.mr * 100
    per_class = {k: round(v, 4) for k, v in zip(metrics.names.values(), metrics.box.maps)}
    emit([
        "",
        f"mAP50: {metrics.box.map50:.4f}  mAP50-95: {metrics.box.map:.4f}",
        f"Precision: {p:.1f} %  Recall: {r:.1f} %",
        f"per-class mAP50-95: {per_class}",
        "",
    ])

    eff = _compute_efficiency_metrics(
        model.model, args.imgsz, select_device(args.device), ckpt_path=args.weights
    )
    emit(_format_efficiency_report(eff))

    size_mb = eff["file_size_mb"] if eff.get("file_size_mb") is not None else eff["model_size_mb"]
    w = 58
    emit([
        "",
        "-" * w,
        "  PAPER TABLE ROW (copy this line)",
        "-" * w,
        "  P(%), R(%), mAP50, mAP50-95, Params(M), GFLOPs, Size(MB), FPS, Epochs, imgsz",
        f"  {p:.1f}, {r:.1f}, {metrics.box.map50:.4f}, {metrics.box.map:.4f}, "
        f"{eff['params_m']:.3f}, {eff['gflops']:.2f}, {size_mb:.2f}, {eff['fps']:.1f}, "
        f"{train_args.get('epochs', '?')}, {args.imgsz}",
    ])

    if not args.no_log:
        path = log_path_for(run_name)
        append_log(path, report)
        print(f"\n[log] da ghi them vao {path}")


if __name__ == "__main__":
    main()
