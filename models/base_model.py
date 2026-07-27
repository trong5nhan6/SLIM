"""Abstract base class for every idea (baseline, lwso, ...).

Each idea subclasses BaseModel and implements build(), which must set self._yolo
to an ultralytics.YOLO instance. train.py stays idea-agnostic: it goes through
build_model() (models/__init__.py) + BaseModel.train(), never branching on
`idea` itself. To add a new idea, add one file here — see baseline.py/lwso.py.

Kept import-light at module level (no torch/ultralytics/lwso here) so that
`from models import MODEL_REGISTRY` stays cheap — train.py needs it just to
build --idea's argparse choices, before any heavy import happens.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_WIDTH = 58


def log_path_for(name: str) -> Path:
    """<repo>/logs/<name>.log, parent created. Absolute so the file lands in the same
    place no matter what cwd train.py/val.py were launched from."""
    p = REPO_ROOT / "logs" / f"{name}.log"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def append_log(path, lines, echo: bool = False) -> None:
    """Append lines to a log file; optionally mirror them to stdout."""
    text = "\n".join(lines)
    if echo:
        print(text)
    with open(path, "a", encoding="utf-8") as f:
        f.write(text + "\n")


def log_row(label: str, value) -> str:
    return f"  {label:<32} {value}"


def log_banner(title: str) -> list[str]:
    """Section header, timestamped so several runs appending to one file stay readable."""
    from datetime import datetime

    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return ["", "=" * LOG_WIDTH, f"  {title}", f"  {stamp}", "=" * LOG_WIDTH]


class BaseModel(ABC):
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self._yolo = None
        self.build()

    @abstractmethod
    def build(self) -> None:
        """Set self._yolo (an ultralytics.YOLO instance) for this idea."""
        ...

    def _build_yolo(self):
        """Shared cfg.model_cfg (+ optional cfg.weights warm-start) loading.

        Call this from build() after any idea-specific setup that must happen
        before YOLO(...) is constructed (e.g. register_lwso() for a model .yaml
        that references custom modules).
        """
        from ultralytics import YOLO

        yolo = YOLO(str(self.cfg.model_cfg))
        if self.cfg.get("weights"):
            yolo.load(str(self.cfg.weights))
        return yolo

    def get_callbacks(self) -> dict[str, Callable]:
        """Extra ultralytics callbacks to register before train().

        Default: write logs/<cfg.name>.log -- a model-info header, one line per epoch,
        a periodic test-split + efficiency block (every cfg.test_every epochs, 0 disables
        just that part), and a closing summary. Shared by every idea since none of it
        depends on architecture/loss. To add idea-specific callbacks in a subclass, merge
        with this dict rather than replacing it:
            def get_callbacks(self):
                return {**super().get_callbacks(), "on_train_epoch_end": my_cb}
        """
        test_every = int(self.cfg.get("test_every", 0) or 0)
        path = log_path_for(str(self.cfg.name))
        print(f"[log] train/test log -> {path}")
        if test_every <= 0:
            print("[log] test_every=0 -> chi ghi header + per-epoch + summary, "
                  "khong co block test-split/efficiency")
        return _build_log_callbacks(self.cfg, self._yolo, test_every, path)

    def train(self, **train_kwargs) -> None:
        assert self._yolo is not None, "build() must run before train()"
        from models._patch_utils import patch_ddp_registration

        patch_ddp_registration()  # no-op unless device has >1 GPU (e.g. "0,1")
        for event, cb in self.get_callbacks().items():
            self._yolo.add_callback(event, cb)
        self._yolo.train(**train_kwargs)

    @property
    def yolo(self):
        return self._yolo


def _compute_efficiency_metrics(
    model, imgsz: int, device, warmup: int = 10, runs: int = 30, ckpt_path=None
) -> dict:
    """Params/GFLOPs/size + batch=1 latency/FPS for `model` on `device`.

    Two size numbers, because they are NOT interchangeable and papers report the second:
      * `model_size_mb`  = params*4 bytes, the fp32 theoretical size. Always available.
      * `file_size_mb`   = os.path.getsize(ckpt_path), the real checkpoint on disk, only
        when `ckpt_path` is given (the training callback has no file to stat). Ultralytics
        saves checkpoints in fp16, so this lands at roughly HALF the fp32 number -- quoting
        the fp32 one next to a paper's reported weight-file size makes our model look twice
        as heavy as it is.

    Latency/FPS come from real forward passes (warmup then timed loop), matching how
    yolov12n-visdrone's utils.metrics.compute_efficiency_metrics measures it, for
    cross-project comparability.
    """
    import time

    import torch
    from ultralytics.utils.torch_utils import get_flops, get_num_params

    dev = device if isinstance(device, torch.device) else torch.device(device)
    params = get_num_params(model)
    gflops = get_flops(model, imgsz)

    dummy = torch.zeros(1, 3, imgsz, imgsz, device=dev)
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(dummy)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(runs):
            model(dummy)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

    latency_ms = (t1 - t0) / runs * 1000
    file_size_mb = None
    if ckpt_path is not None:
        import os

        try:
            file_size_mb = os.path.getsize(str(ckpt_path)) / 1e6
        except OSError:
            file_size_mb = None  # path gone/unreadable -- fp32 number still reported

    return {
        "params_m": params / 1e6,
        "gflops": gflops,
        "model_size_mb": params * 4 / 1e6,  # fp32 theoretical
        "file_size_mb": file_size_mb,  # real .pt on disk (fp16), None if no path given
        "latency_ms": latency_ms,
        "fps": 1000 / latency_ms if latency_ms > 0 else float("inf"),
        "device": str(dev),
    }


def _format_efficiency_report(eff: dict) -> list[str]:
    w = 58
    row = lambda label, value: f"  {label:<32} {value}"  # noqa: E731
    lines = [
        "-" * w,
        "  EFFICIENCY METRICS",
        "-" * w,
        row("Parameters", f"{eff['params_m']:.3f} M"),
        row("GFLOPs", f"{eff['gflops']:.2f} G"),
        row("Model size (fp32, theoretical)", f"{eff['model_size_mb']:.2f} MB"),
    ]
    if eff.get("file_size_mb") is not None:
        # The number to put in a paper table -- same quantity other papers quote.
        lines.append(row("Checkpoint file size (.pt)", f"{eff['file_size_mb']:.2f} MB"))
    lines += [
        row("Latency (batch=1)", f"{eff['latency_ms']:.2f} ms/img"),
        row("FPS (batch=1)", f"{eff['fps']:.1f} fps"),
        row("Device", eff["device"]),
    ]
    return lines


def _format_epoch_line(trainer, epoch: int) -> str:
    """One-line per-epoch trace: train loss + val metrics (already computed by
    ultralytics' own per-epoch validate() by the time on_fit_epoch_end fires) + lr.
    File-only (not printed) -- ultralytics already prints an equivalent progress
    table to console every epoch, so this would just duplicate it there.
    """
    tloss = trainer.tloss
    names = getattr(trainer, "loss_names", ("box_loss", "cls_loss", "dfl_loss"))
    if tloss is not None:
        vals = tloss.tolist() if hasattr(tloss, "tolist") else list(tloss)
        loss_str = "  ".join(f"{n}={v:.4f}" for n, v in zip(names, vals))
    else:
        loss_str = "loss=N/A"

    metrics = trainer.metrics or {}

    def _fmt(v):
        return f"{v:.4f}" if v is not None else " N/A"

    metric_str = (
        f"P={_fmt(metrics.get('metrics/precision(B)'))} "
        f"R={_fmt(metrics.get('metrics/recall(B)'))} "
        f"mAP50={_fmt(metrics.get('metrics/mAP50(B)'))} "
        f"mAP50-95={_fmt(metrics.get('metrics/mAP50-95(B)'))}"
    )
    lr_vals = list((trainer.lr or {}).values())
    lr_str = f"lr={lr_vals[0]:.6f}" if lr_vals else "lr=N/A"

    total = getattr(trainer, "epochs", "?")
    return f"[EPOCH {epoch:>4}/{total}]  {loss_str}  |  {metric_str}  |  {lr_str}"


def _format_model_info(cfg, model) -> list[str]:
    """The 'what was this run' block: which config/architecture, how big, what recipe.

    Params/GFLOPs are measured on the real built model rather than copied from a comment,
    and no latency benchmark runs here -- at on_train_start the model may still be on CPU,
    and the meaningful device-side number is already produced by the periodic
    EFFICIENCY METRICS block below.
    """
    from ultralytics.utils.torch_utils import get_flops, get_num_params

    imgsz = int(cfg.imgsz)
    nwd = (
        f"CIoU-NWD blend (ratio={cfg.get('nwd_ratio', 0.5)}, C={cfg.get('nwd_constant', 12.8)})"
        if cfg.get("use_nwd")
        else "CIoU only (no NWD)"
    )
    params = get_num_params(model)
    return [
        log_row("Config", cfg.get("config_file", "(--idea, khong qua --config)")),
        log_row("Idea", cfg.idea),
        log_row("Model cfg", cfg.model_cfg),
        # Unfused count on purpose: fusing happens later, and ultralytics' own "N layers"
        # line is the FUSED number -- labelling this avoids two different counts in one log.
        log_row("Layers (unfused)", len(list(model.modules()))),
        log_row("Parameters", f"{params / 1e6:.3f} M"),
        log_row("GFLOPs", f"{get_flops(model, imgsz):.2f} G @ imgsz {imgsz}"),
        log_row("Loss", nwd),
        log_row("Data", cfg.data),
        log_row(
            "Recipe",
            f"epochs={cfg.epochs} imgsz={imgsz} batch={cfg.batch} "
            f"seed={cfg.get('seed')} device={cfg.get('device')}",
        ),
    ]


def _build_log_callbacks(cfg, yolo, every: int, path) -> dict[str, Callable]:
    """Callbacks that write logs/<cfg.name>.log for one training run.

    Layout of the file:
        RUN <name> banner + model info      (on_train_start)
        [EPOCH n/N] one line per epoch      (on_fit_epoch_end, always)
        test-split mAP + EFFICIENCY METRICS (on_fit_epoch_end, every `every` epochs)
        SUMMARY banner                      (on_train_end)
    val.py later appends its own EVAL blocks to the SAME file, so one config ends up with
    one log holding train, test and model info together.

    The periodic block uses a standalone DetectionValidator (model=... path, not
    trainer=...) so it never touches trainer.validator / trainer.stopper / best.pt
    selection, which stay driven by the ordinary val split. Test mAP here is monitoring
    only, not model selection (test-dev has no official public labels).

    It validates a deepcopy of the EMA weights, not the live trainer.ema.ema reference:
    AutoBackend fuses whatever nn.Module it's handed (in place, `model.fuse()`), which
    permanently changes its state_dict keys. Handing it the live EMA model corrupts it and
    crashes the next `ema.update(self.model)` call with a KeyError. The deepcopy is reused
    for the efficiency benchmark too (fusion there is harmless, and representative of real
    deployment latency) since it is discarded at the end of the callback either way.
    """
    import copy

    from ultralytics.models.yolo.detect import DetectionValidator
    from ultralytics.utils import RANK

    imgsz, batch = int(cfg.imgsz), int(cfg.batch)
    state = {"validator": None, "csv_path": None, "last_epoch": None}

    def _log(lines, echo: bool = False) -> None:
        append_log(path, lines if isinstance(lines, list) else [lines], echo=echo)

    def _on_train_start(trainer):
        if RANK not in (-1, 0):
            return
        _log(log_banner(f"RUN {cfg.name}"))
        _log(_format_model_info(cfg, trainer.model))
        _log([log_row("Output dir", trainer.save_dir), ""])

    def _on_fit_epoch_end(trainer):
        if RANK not in (-1, 0):
            return
        epoch = trainer.epoch + 1  # 1-indexed, matches the printed epoch column
        # trainer.final_eval() re-fires on_fit_epoch_end once more at the same epoch
        # after training ends; skip the repeat instead of double-logging/re-validating.
        if epoch == state["last_epoch"]:
            return
        state["last_epoch"] = epoch

        _log(_format_epoch_line(trainer, epoch))  # file only -- ultralytics prints its own

        if every <= 0 or epoch % every != 0:
            return

        if state["validator"] is None:
            state["validator"] = DetectionValidator(
                args=dict(
                    data=str(cfg.data),
                    split="test",
                    imgsz=imgsz,
                    batch=batch,
                    plots=False,
                    save_json=False,
                    device=trainer.device,
                ),
                save_dir=trainer.save_dir / "test_eval",
            )
            state["csv_path"] = trainer.save_dir / "test_metrics.csv"
            if not state["csv_path"].exists():
                state["csv_path"].write_text(
                    "epoch,mAP50,mAP50-95,params_m,gflops,model_size_mb,latency_ms,fps\n"
                )
            print(f"[log] test-set eval csv -> {state['csv_path']}")

        _log([f"", f"[test-set eval @ epoch {epoch}]"], echo=True)
        model = copy.deepcopy(trainer.ema.ema or trainer.model)
        stats = state["validator"](model=model)
        map50 = stats.get("metrics/mAP50(B)", float("nan"))
        map5095 = stats.get("metrics/mAP50-95(B)", float("nan"))
        _log([log_row("test mAP50", f"{map50:.4f}"),
              log_row("test mAP50-95", f"{map5095:.4f}")], echo=True)

        eff = _compute_efficiency_metrics(model, imgsz, trainer.device)
        _log(_format_efficiency_report(eff), echo=True)

        with open(state["csv_path"], "a") as f:
            f.write(
                f"{epoch},{map50:.5f},{map5095:.5f},{eff['params_m']:.5f},{eff['gflops']:.5f},"
                f"{eff['model_size_mb']:.5f},{eff['latency_ms']:.5f},{eff['fps']:.5f}\n"
            )

    def _on_train_end(trainer):
        if RANK not in (-1, 0):
            return
        best_fit = getattr(trainer, "best_fitness", None)
        _log(log_banner(f"SUMMARY {cfg.name}"))
        _log([
            log_row("Epochs completed", state["last_epoch"] or "?"),
            log_row("Best fitness", f"{float(best_fit):.5f}" if best_fit is not None else "N/A"),
            log_row("Best weights", getattr(trainer, "best", "?")),
            log_row("Output dir", trainer.save_dir),
            "",
            "  Chay val.py voi best.pt de ghi tiep block EVAL (val + test) vao chinh file nay.",
            "",
        ])

    return {
        "on_train_start": _on_train_start,
        "on_fit_epoch_end": _on_fit_epoch_end,
        "on_train_end": _on_train_end,
    }
