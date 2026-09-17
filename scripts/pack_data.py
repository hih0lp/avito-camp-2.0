"""Упаковывает внешние датасеты в data/packed/ (см. src/packed.py).

Все картинки считаются нормально стоящими: это кропы из датасетов для распознавания текста.
Метку «перевёрнут» создаём сами при обучении и валидации.

Источники (скачаны с HuggingFace, см. SOLUTION.md):
  printed6     DonkeySmall/OCR-Cyrillic-Printed-6   синтетическая печатная кириллица (берём подвыборку)
  printed10    DonkeySmall/OCR-Cyrillic-Printed-10  то же, другой генератор
  union_train  Bekhouche/Union14M-L-STR (2 train-файла)  реальные фото, в основном латиница
  union_valid  Bekhouche/Union14M-L-STR (valid-00000)    реальная валидация
  ru_plates    Foximaz/russian_ocr_small car_plate       реальные номера машин (валидация)
  ru_hand      Foximaz/russian_ocr_small handwriting     рукописная кириллица (валидация)
  synth_train / synth_val — наша синтетика (src/synth.py)

    python pack_data.py --only printed6 union_train
"""
import argparse
import io
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1] # корень проекта
sys.path.insert(0, str(PROJECT))

from src.packed import write_packed # noqa: E402

DATA = PROJECT / "data"
OUT = DATA / "packed"
SEED = 42

# Кропы, где высота заметно больше ширины, часто повёрнуты на 90° или содержат вертикальный текст.
# В тесте таких 4 из 20 000, поэтому из обучения их убираем, чтобы не было шумных меток.
MAX_H_OVER_W = 1.2


def image_size(data: bytes) -> tuple[int, int]:
    with Image.open(io.BytesIO(data)) as im:
        return im.size


def from_zip(path: Path, limit: int, seed: int):
    with zipfile.ZipFile(path) as zf:
        txt = next(n for n in zf.namelist() if n.endswith(".txt"))
        # формат строк: "6/image_6_0.jpg`текст"
        lines = zf.read(txt).decode("utf-8-sig").splitlines()
        rows = [ln.split("`", 1) for ln in lines if "`" in ln]
        rng = np.random.default_rng(seed)
        pick = rng.choice(len(rows), size=min(limit, len(rows)), replace=False)
        for k in np.sort(pick):
            name, text = rows[k]
            yield zf.read(name), {"source": path.parent.name, "name": name, "text": text}


def from_parquet(paths: list[Path], limit: int | None, seed: int, drop_difficulty=()):
    items = []
    for path in paths:
        pf = pq.ParquetFile(path)
        for g in range(pf.num_row_groups):
            table = pf.read_row_group(g).to_pandas()
            for r in table.itertuples(index=False):
                if drop_difficulty and getattr(r, "difficulty", None) in drop_difficulty:
                    continue
                data = r.image["bytes"]
                w, h = image_size(data)
                if h > MAX_H_OVER_W * w:
                    continue
                items.append((data, {"source": getattr(r, "source_dataset", path.stem), "text": r.text}))
    if limit and len(items) > limit:
        rng = np.random.default_rng(seed)
        items = [items[k] for k in np.sort(rng.choice(len(items), size=limit, replace=False))]
    yield from items


def from_dir(name: str):
    df = pd.read_csv(DATA / name / "labels.csv")
    for r in df.itertuples():
        data = (DATA / name / "images" / f"{r.image_id}.png").read_bytes()
        if r.label: # наша синтетика уже частично перевёрнута — храним нормальную ориентацию
            img = cv2.rotate(cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR), cv2.ROTATE_180)
            data = cv2.imencode(".png", img)[1].tobytes()
        yield data, {"source": name, "text": r.text}


SOURCES = {
    "printed6": lambda: from_zip(DATA / "ocr_cyrillic_printed6/data_6.zip", 300_000, SEED),
    "printed10": lambda: from_zip(DATA / "ocr_cyrillic_printed10/data_10.zip", 300_000, SEED + 1),
    # hard/challenging в Union14M часто кривые/повёрнутые — для задачи ориентации это шум
    "union_train": lambda: from_parquet(sorted((DATA / "union14m_l").glob("train-*.parquet")), None, SEED,
                                        drop_difficulty=("hard", "challenging")),
    "union_valid": lambda: from_parquet([DATA / "union14m_l/valid-00000-of-00003.parquet"], 5000, SEED,
                                        drop_difficulty=("hard", "challenging")),
    "ru_plates": lambda: from_parquet([DATA / "russian_ocr_small/car_plate.parquet"], None, SEED),
    "ru_hand": lambda: from_parquet([DATA / "russian_ocr_small/handwriting.parquet"], None, SEED),
    "synth_train": lambda: from_dir("synth_train"),
    "synth_val": lambda: from_dir("synth_val"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=list(SOURCES))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    for name in args.only:
        if (OUT / f"{name}.idx.npy").exists() and not args.force:
            print(f"{name}: уже упакован")
            continue
        n = write_packed(OUT / name, SOURCES[name]())
        print(f"{name}: {n} картинок", flush=True)


if __name__ == "__main__":
    main()
