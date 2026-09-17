"""Считает признаки одного метода на val (синтетика) или test и сохраняет в features/<split>_<method>.csv.

Paddle на macOS однопоточный, поэтому CPU-методы параллелим процессами (--workers).
Время на кроп пишем в features/timings.csv: меряется внутри воркера, без чтения файлов.

Пример:
    python compute_features.py --method rec_eslav_v5 --split test --workers 4
"""
import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1] # корень проекта
sys.path.insert(0, str(PROJECT))

from src.io_utils import read_image, resolve_ids, resolve_path # noqa: E402
from src.methods import METHODS, build_method # noqa: E402

SPLITS = {
    "val": (PROJECT / "data/synth_val/images", PROJECT / "data/synth_val/labels.csv"),
    "test": (PROJECT / "test/images", PROJECT / "sample_submission.csv"),
    # реальная калибровочная выборка (scripts/make_calib_set.py)
    "calib": (PROJECT / "data/calib_val/images", PROJECT / "data/calib_val/labels.csv"),
}
TORCH_METHODS = {"doctr_crop", "easyocr_ru"}

_method = None


def _init_worker(name: str, device: str):
    global _method
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    _method = build_method(name, device)
    # прогрев, чтобы инициализация не попала в замер времени
    _method([np.full((32, 128, 3), 255, np.uint8)] * 4)


def _aspect(path: Path) -> float:
    with Image.open(path) as img: # читает только заголовок файла
        w, h = img.size
    return w / h


def _process_chunk(args):
    images_dir, ids = args
    images = [read_image(resolve_path(images_dir, i)) for i in ids]
    t0 = time.perf_counter()
    df = _method(images)
    elapsed = time.perf_counter() - t0
    df.insert(0, "image_id", ids)
    return df, elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=METHODS, required=True)
    ap.add_argument("--split", choices=list(SPLITS), required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--device", default="cpu", help="cpu или mps (для torch-методов)")
    ap.add_argument("--limit", type=int, default=0, help="только первые N кропов (для отладки)")
    args = ap.parse_args()

    images_dir, ids_path = SPLITS[args.split]
    ids = resolve_ids(images_dir, ids_path)
    if args.limit:
        ids = ids[: args.limit]
    # Распознаватели приводят батч к ширине самого широкого кропа (паддинг).
    # Если сортировать по w/h, в батч попадают похожие кропы и паддинга почти нет.
    order = sorted(ids, key=lambda i: _aspect(resolve_path(images_dir, i)))
    chunks = [(images_dir, order[i:i + args.chunk]) for i in range(0, len(order), args.chunk)]

    # На MPS один процесс: несколько процессов на одном GPU только мешают друг другу
    workers = 1 if args.device == "mps" else args.workers
    parts, total_model_time = [], 0.0
    t_wall = time.perf_counter()
    with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(args.method, args.device)) as pool:
        for i, (df, elapsed) in enumerate(pool.map(_process_chunk, chunks), 1):
            parts.append(df)
            total_model_time += elapsed
            if i % max(1, len(chunks) // 10) == 0 or i == len(chunks):
                print(f"[{args.method}/{args.split}] {i}/{len(chunks)} чанков, "
                      f"{time.perf_counter() - t_wall:.0f} c", flush=True)
    wall = time.perf_counter() - t_wall

    # Возвращаем исходный порядок id
    out = pd.concat(parts).set_index("image_id").loc[ids].reset_index()
    out_path = PROJECT / "features" / f"{args.split}_{args.method}.csv"
    out_path.parent.mkdir(exist_ok=True)
    out.to_csv(out_path, index=False)

    # Каждый кроп прогоняется дважды (x и rot180(x)); ms_per_pass — на один прогон одного кропа
    timing = {
        "method": args.method, "split": args.split, "n": len(ids), "device": args.device,
        "workers": workers, "wall_s": round(wall, 1),
        "ms_per_pass_single_core": round(1000 * total_model_time / (2 * len(ids)), 3),
    }
    timings_path = PROJECT / "features" / "timings.csv"
    log = pd.DataFrame([timing])
    if timings_path.exists():
        log = pd.concat([pd.read_csv(timings_path), log], ignore_index=True)
    log.to_csv(timings_path, index=False)
    print(f"Сохранено {out_path} | {timing}")


if __name__ == "__main__":
    main()
