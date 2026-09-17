"""Реальная калибровочная выборка для ансамбля: data/calib_val/{images, labels.csv}.

Берём кропы, которые НЕ участвуют в обучении CNN:
  union_valid (реальные фото, 2000 шт.), ru_plates (номера, 1500), ru_hand (рукопись, 1544).
Каждый кроп с вероятностью 0.5 переворачиваем (label=1). Формат как у data/synth_val,
поэтому scripts/compute_features.py считает на ней признаки без изменений (--split calib).

    python make_calib_set.py
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1] # корень проекта
sys.path.insert(0, str(PROJECT))

from src.packed import PackedImages # noqa: E402

PACKED = PROJECT / "data/packed"
OUT = PROJECT / "data/calib_val"
PARTS = {"union_valid": 2000, "ru_plates": None, "ru_hand": None}
SEED = 123


def main():
    if not (PACKED / "union_valid.idx.npy").exists():
        raise SystemExit("Нет упакованных внешних данных (data/packed). Сначала включите DOWNLOAD_EXTERNAL "
                         "в ноутбуке — он скачает датасеты и вызовет scripts/pack_data.py.")
    rng = np.random.default_rng(SEED)
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    rows = []
    for name, limit in PARTS.items():
        pack = PackedImages(PACKED / name)
        meta = pack.meta()
        idx = np.arange(len(pack)) if limit is None else np.sort(rng.choice(len(pack), limit, replace=False))
        for i in idx:
            img = pack.get(int(i))
            label = int(rng.random() < 0.5)
            if label:
                img = cv2.rotate(img, cv2.ROTATE_180)
            image_id = f"calib_{len(rows):05d}"
            cv2.imwrite(str(OUT / "images" / f"{image_id}.png"), img)
            rows.append({"image_id": image_id, "label": label, "text": meta.text[i], "font": name})
    # колонка font — для совместимости с synth_val; здесь в ней имя источника
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "labels.csv", index=False)
    print(df.groupby("font").label.agg(["count", "mean"]))


if __name__ == "__main__":
    main()
