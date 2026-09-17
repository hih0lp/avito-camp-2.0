"""Обучение CNN (src/model.py) на внешних датасетах. Тестовые картинки в обучении НЕ используются.

Обучение: смесь упакованных источников (pack_data.py), за эпоху — фиксированное число картинок из каждого.
Валидация на каждой эпохе (кропы с известной ориентацией, половина перевёрнута):
  union_valid — реальные фото (латиница), ru_plates — реальные номера, ru_hand — рукописная кириллица,
  synth_val — наша синтетика (в обучение не входит).
Лучший чекпоинт — по среднему Brier на валидациях.

    python train.py --tag cnn_v1 --epochs 12
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT = Path(__file__).resolve().parents[1] # корень проекта
PACKED = PROJECT / "data/packed"
sys.path.insert(0, str(PROJECT))

from src.data import FlipEvalDataset, MixtureTrainDataset # noqa: E402
from src.device import best_device # noqa: E402
from src.model import build_model, count_params # noqa: E402

TRAIN_SOURCES = {"printed6": 120_000, "printed10": 120_000, "union_train": 160_000}
EVAL_SOURCES = ["union_valid", "ru_plates", "ru_hand", "synth_val"]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        probs.append(torch.sigmoid(model(x.to(device))).float().cpu())
        labels.append(y)
    p, y = torch.cat(probs).numpy(), torch.cat(labels).numpy()
    return {"brier": float(np.mean((p - y) ** 2)), "acc": float(np.mean((p > 0.5) == y))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="cnn_v1")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--scale", type=float, default=1.0, help="множитель к числу картинок за эпоху")
    ap.add_argument("--arch", default="cnn", help="cnn или имя модели timm (mobilenetv3_small_050, lcnet_050)")
    ap.add_argument("--pretrained", action="store_true", help="веса ImageNet для timm-модели")
    ap.add_argument("--height", type=int, default=32)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = best_device()

    sources = {name: (PACKED / name, int(n * args.scale)) for name, n in TRAIN_SOURCES.items()
               if (PACKED / f"{name}.idx.npy").exists()}
    size = (args.height, args.width)
    train_ds = MixtureTrainDataset(sources, seed=args.seed, size=size)
    eval_loaders = {
        name: DataLoader(FlipEvalDataset(PACKED / name, seed=0, size=size), batch_size=512, num_workers=4)
        for name in EVAL_SOURCES if (PACKED / f"{name}.idx.npy").exists()
    }

    model = build_model(args.arch, pretrained=args.pretrained, dropout_p=args.dropout).to(device)
    print(f"Параметров: {count_params(model):,}; источники: { {k: v[1] for k, v in sources.items()} }; "
          f"валидация: {list(eval_loaders)}; device: {device}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps_per_epoch = len(train_ds) // args.batch
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr,
                                                    total_steps=args.epochs * steps_per_epoch)
    criterion = nn.BCEWithLogitsLoss()

    ckpt_dir = PROJECT / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    (ckpt_dir / f"{args.tag}_config.json").write_text(
        json.dumps({**vars(args), "train_sources": {k: v[1] for k, v in sources.items()}}, indent=2))
    history, best = [], float("inf")
    for epoch in range(args.epochs):
        train_ds.set_epoch(epoch) # воркеры не persistent: новая выборка доходит до них
        gen = torch.Generator().manual_seed(args.seed + epoch)
        loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, generator=gen, drop_last=True,
                            num_workers=args.workers, persistent_workers=False, prefetch_factor=4)
        model.train()
        t0, total, n = time.perf_counter(), 0.0, 0
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            # fp16 на GPU (CUDA/MPS) ускоряет обучение; лосс считаем в float32
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type in ("cuda", "mps")):
                logits = model(x)
            loss = criterion(logits.float(), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total += loss.item() * len(y)
            n += len(y)

        row = {"epoch": epoch + 1, "train_loss": total / n, "time_s": round(time.perf_counter() - t0, 1)}
        for name, eval_loader in eval_loaders.items():
            for k, v in evaluate(model, eval_loader, device).items():
                row[f"{name}_{k}"] = v
        row["mean_val_brier"] = float(np.mean([row[f"{k}_brier"] for k in eval_loaders]))
        history.append(row)
        pd.DataFrame(history).to_csv(ckpt_dir / f"{args.tag}_history.csv", index=False)
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}), flush=True)

        if row["mean_val_brier"] <= best:
            best = row["mean_val_brier"]
            torch.save(model.state_dict(), ckpt_dir / f"{args.tag}.pt")

    print(f"Лучший средний val Brier: {best:.5f}; чекпоинт {ckpt_dir / (args.tag + '.pt')}")


if __name__ == "__main__":
    main()
