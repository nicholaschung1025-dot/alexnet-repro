"""Training and evaluation loops."""

from __future__ import annotations

from contextlib import nullcontext

import torch


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count else 0.0


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1, 5)):
    maxk = min(max(topk), output.size(1))
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    result = []
    for k in topk:
        k = min(k, maxk)
        correct_k = correct[:k].reshape(-1).float().sum(0)
        result.append(correct_k.mul_(100.0 / target.size(0)).item())
    return result


def _autocast(device: torch.device, amp: bool):
    if amp and device.type == "cuda":
        return torch.amp.autocast("cuda")
    return nullcontext()


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device: torch.device,
    amp: bool = True,
    grad_clip: float = 0.0,
    limit_batches: int | None = None,
) -> dict:
    model.train()
    loss_meter = AverageMeter()
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()

    for batch_idx, (images, targets) in enumerate(loader):
        if limit_batches is not None and batch_idx >= limit_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, amp):
            outputs = model(images)
            loss = criterion(outputs, targets)

        if scaler is not None and scaler.is_enabled():
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

        top1, top5 = accuracy(outputs.detach(), targets, topk=(1, 5))
        n = images.size(0)
        loss_meter.update(loss.item(), n)
        top1_meter.update(top1, n)
        top5_meter.update(top5, n)

    return {
        "loss": loss_meter.avg,
        "top1": top1_meter.avg,
        "top5": top5_meter.avg,
    }


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device: torch.device,
    amp: bool = True,
    limit_batches: int | None = None,
) -> dict:
    model.eval()
    loss_meter = AverageMeter()
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()

    for batch_idx, (images, targets) in enumerate(loader):
        if limit_batches is not None and batch_idx >= limit_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # 10-crop evaluation stacks crops in the batch dimension.
        ten_crop = images.ndim == 5
        if ten_crop:
            batch, crops, channels, height, width = images.shape
            images = images.reshape(batch * crops, channels, height, width)

        with _autocast(device, amp):
            outputs = model(images)
            if ten_crop:
                outputs = outputs.reshape(batch, crops, -1).softmax(dim=-1).mean(dim=1)
                loss = -outputs.gather(1, targets.view(-1, 1)).clamp_min(1e-12).log().mean()
            else:
                loss = criterion(outputs, targets)

        top1, top5 = accuracy(outputs, targets, topk=(1, 5))
        n = targets.size(0)
        loss_meter.update(loss.item(), n)
        top1_meter.update(top1, n)
        top5_meter.update(top5, n)

    return {
        "loss": loss_meter.avg,
        "top1": top1_meter.avg,
        "top5": top5_meter.avg,
    }
