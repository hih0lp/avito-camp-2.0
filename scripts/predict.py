"""Инференс CNN: признаки (p_x, p_rot) для val / calib / test + замер скорости.

    python predict.py --tag cnn_v1
Пишет features/{split}_{tag}.csv и features/speed_{tag}.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT = Path(__file__).resolve().parents[1] # корень проекта
sys.path.insert(0, str(PROJECT))

from src.data import FileDataset, rot180 # noqa: E402
from src.device import best_device # noqa: E402
from src.io_utils import resolve_ids # noqa: E402
from src.model import build_model, count_params # noqa: E402

SPLITS = {
    "val": (PROJECT / "data/synth_val/images", PROJECT / "data/synth_val/labels.csv"),
    "calib": (PROJECT / "data/calib_val/images", PROJECT / "data/calib_val/labels.csv"),
    "test": (PROJECT / "test/images", PROJECT / "sample_submission.csv"),
}


def load_config(tag: str) -> dict:
    return json.loads((PROJECT / "checkpoints" / f"{tag}_config.json").read_text())


def load_model(tag: str, device: str) -> torch.nn.Module:
    cfg = load_config(tag)
    model = build_model(cfg.get("arch", "cnn"), dropout_p=cfg.get("dropout", 0.3))
    model.load_state_dict(torch.load(PROJECT / "checkpoints" / f"{tag}.pt", map_location="cpu"))
    return model.to(device).eval()


@torch.no_grad()
def predict_split(model, split: str, device: str, size: tuple[int, int]) -> pd.DataFrame:
    images_dir, ids_path = SPLITS[split]
    ids = resolve_ids(images_dir, ids_path)
    loader = DataLoader(FileDataset([images_dir / f"{i}.png" for i in ids], size=size), batch_size=512, num_workers=6)
    p_x, p_rot = [], []
    for x in loader:
        x = x.to(device)
        p_x.append(torch.sigmoid(model(x)).cpu())
        p_rot.append(torch.sigmoid(model(rot180(x))).cpu()) # по построению = 1 - p_x
    return pd.DataFrame({"image_id": ids, "p_x": torch.cat(p_x).numpy(), "p_rot": torch.cat(p_rot).numpy()})


@torch.no_grad()
def benchmark(model_cpu, size: tuple[int, int], n: int = 512) -> dict:
    """Скорость на CPU в один поток (как «прод») и в батче."""
    torch.set_num_threads(1)
    x = torch.randn(n, 3, *size)
    out = {}
    for bs in (1, 64):
        for _ in range(3):
            model_cpu(x[:bs])
        t0 = time.perf_counter()
        for i in range(0, n, bs):
            model_cpu(x[i:i + bs])
        out[f"cpu1_bs{bs}_ms_per_crop"] = round(1000 * (time.perf_counter() - t0) / n, 3)
    torch.set_num_threads(torch.get_num_threads())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="cnn_v1")
    ap.add_argument("--splits", nargs="*", default=list(SPLITS))
    args = ap.parse_args()
    device = best_device().type
    out_dir = PROJECT / "features"
    out_dir.mkdir(exist_ok=True)

    cfg = load_config(args.tag)
    size = (cfg.get("height", 32), cfg.get("width", 256))
    model = load_model(args.tag, device)
    for split in args.splits:
        t0 = time.perf_counter()
        df = predict_split(model, split, device, size)
        df.to_csv(out_dir / f"{split}_{args.tag}.csv", index=False)
        print(f"{split}: {len(df)} кропов за {time.perf_counter() - t0:.1f} c", flush=True)

    ckpt = PROJECT / "checkpoints" / f"{args.tag}.pt"
    # Антисимметричная модель уже внутри делает два прогона (x и rot180(x)) — это и есть время на кроп
    speed = {"params": count_params(model), "checkpoint_mb": round(ckpt.stat().st_size / 2**20, 2),
             **benchmark(load_model(args.tag, "cpu"), size)}
    (out_dir / f"speed_{args.tag}.json").write_text(json.dumps(speed, indent=2))
    print(speed)


if __name__ == "__main__":
    main()
