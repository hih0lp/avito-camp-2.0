"""Упакованный датасет: все закодированные картинки подряд в одном .bin + массив смещений .idx.npy.

Сотни тысяч мелких файлов читаются медленно. Один большой файл, открытый через memmap,
читается воркерами DataLoader параллельно и почти без накладных расходов.
"""
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd


def write_packed(out_prefix: Path, items: Iterable[tuple[bytes, dict]]) -> int:
    """items: (закодированная картинка, метаданные). Пишет <prefix>.bin, <prefix>.idx.npy, <prefix>.meta.csv."""
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    offsets, meta = [0], []
    with open(f"{out_prefix}.bin", "wb") as f:
        for data, info in items:
            f.write(data)
            offsets.append(offsets[-1] + len(data))
            meta.append(info)
    np.save(f"{out_prefix}.idx.npy", np.asarray(offsets, dtype=np.int64))
    pd.DataFrame(meta).to_csv(f"{out_prefix}.meta.csv", index=False)
    return len(meta)


class PackedImages:
    def __init__(self, prefix: Path):
        self.prefix = Path(prefix)
        self.offsets = np.load(f"{self.prefix}.idx.npy")
        self._data = None # memmap открываем лениво: в каждом воркере свой

    def __len__(self):
        return len(self.offsets) - 1

    def __getstate__(self):
        # в воркеры передаём только пути и смещения, не открытый memmap
        return {"prefix": self.prefix, "offsets": self.offsets, "_data": None}

    def get(self, i: int) -> np.ndarray:
        if self._data is None:
            self._data = np.memmap(f"{self.prefix}.bin", dtype=np.uint8, mode="r")
        buf = np.asarray(self._data[self.offsets[i]:self.offsets[i + 1]])
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"{self.prefix}: не удалось декодировать #{i}")
        return img

    def meta(self) -> pd.DataFrame:
        return pd.read_csv(f"{self.prefix}.meta.csv")
