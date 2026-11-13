"""Train AlexNet on an ImageFolder dataset."""

from __future__ import annotations

import argparse
import sys
import time
from contextlib import nullcontext
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau

from src.data import build_dataloaders
from src.engine import evaluate, train_one_epoch
from src.model import AlexNet, count_parameters
from src.utils import (
    CSVLogger,
    ensure_dir,
    get_device,
    load_checkpoint,
    load_config,
    save_checkpoint,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AlexNet from scratch.")
    parser.add_argument("--config", required=True, help="Path to a YAML config.")
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default=None,
        help="Checkpoint path, or omit the value to use <output.dir>/last.pth.",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--limit-batches", type=int, default=None)
    parser.add_argument(
        "--overfit-one-batch",
        action="store_true",
        help="Sanity check: overfit a single random batch, no dataset required.",
    )
    parser.add_argument("--overfit-steps", type=int, default=50)
    return parser.parse_args()


def make_scaler(enabled: bool):
    if not enabled:
        return None
    try:
        return torch.amp.GradScaler("cuda")
    except TypeError:
        return torch.cuda.amp.GradScaler()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    seed = args.seed if args.seed is not None else int(cfg.get("seed", 0))
    set_seed(seed)
    device = get_device()
    if device.type != "cuda":
        print("WARNING: CUDA is not available; running on CPU.")

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]

    num_classes = int(data_cfg["num_classes"])
    model = AlexNet(
        num_classes=num_classes,
        dropout=float(model_cfg.get("dropout", 0.5)),
        activation=str(model_cfg.get("activation", "relu")),
        use_lrn=bool(model_cfg.get("use_lrn", True)),
        init=str(model_cfg.get("init", "paper")),
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        label_smoothing=float(train_cfg.get("label_smoothing", 0.0))
    )
    optimizer = SGD(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        momentum=float(train_cfg.get("momentum", 0.9)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0005)),
    )

    milestones = train_cfg.get("lr_milestones") or []
    scheduler_kind = str(train_cfg.get("scheduler", "multistep"))
    if scheduler_kind == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.1,
            patience=int(train_cfg.get("lr_patience", 3)),
            min_lr=float(train_cfg.get("min_lr", 1e-5)),
        )
    elif milestones:
        scheduler = MultiStepLR(
            optimizer, milestones=[int(m) for m in milestones], gamma=0.1
        )
    else:
        scheduler = None

    amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    scaler = make_scaler(amp)

    print(f"Device: {device}")
    print(f"Model parameters: {count_parameters(model):,}")
    print(f"Classes: {num_classes}")

    if args.overfit_one_batch:
        batch_size = min(8, int(data_cfg.get("batch_size", 8)))
        images = torch.randn(batch_size, 3, 224, 224, device=device)
        targets = torch.randint(0, num_classes, (batch_size,), device=device)
        model.train()
        grad_clip = float(train_cfg.get("grad_clip", 0.0))
        print(f"Overfitting one batch: batch={batch_size}, steps={args.overfit_steps}")
        for step in range(args.overfit_steps):
            optimizer.zero_grad(set_to_none=True)
            context = torch.amp.autocast("cuda", enabled=amp) if amp else nullcontext()
            with context:
                outputs = model(images)
                loss = criterion(outputs, targets)
            if scaler is not None:
                scaler.scale(loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            if step % 10 == 0 or step == args.overfit_steps - 1:
                print(f"  step {step:03d} | loss {loss.item():.4f}")
        print("Overfit sanity check finished.")
        return

    train_loader, val_loader, classes = build_dataloaders(cfg)
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    print(f"Classes: {classes}")

    out_dir = ensure_dir(cfg["output"]["dir"])
    with open(out_dir / "config.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)

    epochs = args.epochs if args.epochs is not None else int(train_cfg["epochs"])
    limit_batches = (
        args.limit_batches
        if args.limit_batches is not None
        else train_cfg.get("limit_batches")
    )

    start_epoch = 0
    best_top1 = 0.0
    if args.resume:
        resume_path = out_dir / "last.pth" if args.resume == "auto" else Path(args.resume)
        checkpoint = load_checkpoint(
            resume_path, model, optimizer, scheduler, scaler, device
        )
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_top1 = float(checkpoint.get("best_top1", 0.0))
        print(f"Resumed from {resume_path} at epoch {start_epoch}")

    logger = CSVLogger(
        out_dir / "metrics.csv",
        fieldnames=[
            "epoch",
            "lr",
            "train_loss",
            "train_top1",
            "train_top5",
            "val_loss",
            "val_top1",
            "val_top5",
            "epoch_time",
        ],
    )

    for epoch in range(start_epoch, epochs):
        started = time.time()
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            amp=amp,
            grad_clip=float(train_cfg.get("grad_clip", 0.0)),
            limit_batches=limit_batches,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
            amp=amp,
            limit_batches=limit_batches,
        )
        # Record the LR actually used for this epoch, before stepping the scheduler.
        lr = optimizer.param_groups[0]["lr"]
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_metrics["top1"])
        elif scheduler is not None:
            scheduler.step()

        elapsed = time.time() - started
        row = {
            "epoch": epoch,
            "lr": lr,
            "train_loss": train_metrics["loss"],
            "train_top1": train_metrics["top1"],
            "train_top5": train_metrics["top5"],
            "val_loss": val_metrics["loss"],
            "val_top1": val_metrics["top1"],
            "val_top5": val_metrics["top5"],
            "epoch_time": elapsed,
        }
        logger.log(row)

        print(
            f"Epoch {epoch:03d}/{epochs - 1:03d} | "
            f"lr {lr:.5f} | "
            f"train loss {train_metrics['loss']:.3f} top1 {train_metrics['top1']:.2f} | "
            f"val loss {val_metrics['loss']:.3f} top1 {val_metrics['top1']:.2f} "
            f"top5 {val_metrics['top5']:.2f} | "
            f"{elapsed:.1f}s"
        )

        improved = val_metrics["top1"] > best_top1
        if improved:
            best_top1 = val_metrics["top1"]

        save_checkpoint(
            out_dir / "last.pth",
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            best_top1,
            cfg,
        )
        if improved:
            save_checkpoint(
                out_dir / "best.pth",
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                best_top1,
                cfg,
            )
            print(f"  New best top-1: {best_top1:.2f}%")

    print(f"Training finished. Best val top-1: {best_top1:.2f}%")


if __name__ == "__main__":
    main()
