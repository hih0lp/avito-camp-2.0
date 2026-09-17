"""Данные для CNN: предобработка кропа, аугментации, датасеты.

Все картинки в упакованных источниках стоят нормально. Метку создаём сами:
с вероятностью 0.5 переворачиваем кроп на 180° и ставим label=1.
"""
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .packed import PackedImages

H, W = 32, 256 # медиана w/h в тесте ~4.9, 75-й перцентиль ~7.5; шире 8:1 сжимаем по ширине


def read_bgr(path: Path) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Не удалось прочитать {path}")
    return img


def to_tensor(img: np.ndarray, h: int = H, w: int = W) -> torch.Tensor:
    """BGR-кроп -> тензор 3xHxW в [-1, 1].

    Ресайз по высоте с сохранением пропорций, паддинг нулями ПО ЦЕНТРУ.
    Паддинг по центру важен: rot180 от такого тензора совпадает с предобработкой
    перевёрнутого кропа, поэтому переворот можно делать уже на тензоре.
    """
    ih, iw = img.shape[:2]
    new_w = int(np.clip(round(iw * h / ih), 1, w))
    img = cv2.resize(img, (new_w, h), interpolation=cv2.INTER_AREA if ih > h else cv2.INTER_LINEAR)
    x = torch.from_numpy(img[:, :, ::-1].copy()).permute(2, 0, 1).float() / 127.5 - 1.0
    out = torch.zeros(3, h, w)
    left = (w - new_w) // 2
    out[:, :, left:left + new_w] = x
    return out


def rot180(x: torch.Tensor) -> torch.Tensor:
    return torch.flip(x, dims=(-2, -1))


def _recolor(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Перекраска «тёмный текст на светлом» в случайные цвета текста и фона (+ градиент фона)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0 # 0 — текст, 1 — фон
    h, w = gray.shape
    # фон и текст разводим по яркости минимум на 90, иначе текст становится невидимым
    fg = rng.uniform(0, 255, 3)
    lum = fg.mean()
    target = rng.uniform(lum + 90, 255) if (lum < 165 and (lum < 90 or rng.random() < 0.5)) \
        else rng.uniform(0, lum - 90)
    bg1 = np.clip(rng.normal(target, 25, 3), 0, 255)
    bg2 = np.clip(bg1 + rng.normal(0, 25, 3), 0, 255)
    t = np.linspace(0, 1, w, dtype=np.float32)[None, :, None]
    bg = bg1 * (1 - t) + bg2 * t
    a = gray[:, :, None]
    return (fg * (1 - a) + bg * a).astype(np.float32)


def augment(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Аугментации, не меняющие ориентацию. Горизонтальный флип НЕ используем: зеркальный текст — другая задача."""
    h, w = img.shape[:2]
    # неточный бокс: срезаем/добавляем поля
    if rng.random() < 0.7:
        dx, dy = int(w * rng.uniform(0, 0.08)), int(h * rng.uniform(0, 0.15))
        x0, x1 = rng.integers(0, dx + 1), w - rng.integers(0, dx + 1)
        y0, y1 = rng.integers(0, dy + 1), h - rng.integers(0, dy + 1)
        if x1 - x0 > 4 and y1 - y0 > 4:
            img = img[y0:y1, x0:x1]
    if rng.random() < 0.5:
        pad = int(img.shape[0] * rng.uniform(0, 0.2))
        img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    # лёгкий наклон
    if rng.random() < 0.5:
        h, w = img.shape[:2]
        m = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-5, 5), 1.0)
        img = cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REPLICATE)
    # цвет
    img = _recolor(img, rng) if rng.random() < 0.3 else img.astype(np.float32)
    if rng.random() < 0.8:
        img = img * rng.uniform(0.6, 1.4) + rng.uniform(-40, 40)
    if rng.random() < 0.3:
        img = img[:, :, rng.permutation(3)]
    if rng.random() < 0.2:
        img = 255 - img
    if rng.random() < 0.2:
        img = np.repeat(img.mean(axis=2, keepdims=True), 3, axis=2)
    if rng.random() < 0.3:
        img = img + rng.normal(0, rng.uniform(2, 12), img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    # качество: даунскейл, blur, JPEG
    if rng.random() < 0.3:
        s = rng.uniform(0.3, 0.8)
        small = cv2.resize(img, (max(4, int(img.shape[1] * s)), max(4, int(img.shape[0] * s))))
        img = cv2.resize(small, (img.shape[1], img.shape[0]))
    if rng.random() < 0.3:
        img = cv2.GaussianBlur(img, (3, 3), 0)
    if rng.random() < 0.3:
        q = int(rng.integers(30, 90))
        img = cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)
    return img


class MixtureTrainDataset(Dataset):
    """Смесь упакованных источников. За эпоху из источника i берётся n_i случайных картинок.

    sources: {имя: (prefix, n_per_epoch)}. Выборка индексов и аугментации детерминированы
    по (seed, epoch), поэтому результат воспроизводим при любом числе воркеров.
    """

    def __init__(self, sources: dict[str, tuple[Path, int]], seed: int = 42, size: tuple[int, int] = (H, W)):
        self.size = size
        self.names = list(sources)
        self.packs = [PackedImages(p) for p, _ in sources.values()]
        self.per_epoch = [n for _, n in sources.values()]
        self.seed = seed
        self.set_epoch(0)

    def set_epoch(self, epoch: int):
        self.epoch = epoch
        rng = np.random.default_rng((self.seed, epoch))
        self.index = [] # (номер источника, индекс внутри источника)
        for s, (pack, n) in enumerate(zip(self.packs, self.per_epoch)):
            idx = rng.choice(len(pack), size=n, replace=n > len(pack))
            self.index += [(s, int(i)) for i in idx]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        s, i = self.index[k]
        rng = np.random.default_rng((self.seed, self.epoch, k))
        x = to_tensor(augment(self.packs[s].get(i), rng), *self.size)
        label = int(rng.random() < 0.5)
        if label:
            x = rot180(x)
        return x, torch.tensor(float(label))


class FlipEvalDataset(Dataset):
    """Валидация: упакованный источник, половина кропов перевёрнута (детерминированно по seed)."""

    def __init__(self, prefix: Path, seed: int = 0, limit: int | None = None, size: tuple[int, int] = (H, W)):
        self.size = size
        self.pack = PackedImages(prefix)
        n = len(self.pack)
        rng = np.random.default_rng(seed)
        self.items = np.sort(rng.choice(n, size=min(limit or n, n), replace=False))
        self.labels = (rng.random(len(self.items)) < 0.5).astype(np.float32)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        x = to_tensor(self.pack.get(int(self.items[k])), *self.size)
        if self.labels[k]:
            x = rot180(x)
        return x, torch.tensor(self.labels[k])


class FileDataset(Dataset):
    """Кропы с диска как есть (для инференса на тесте)."""

    def __init__(self, paths, size: tuple[int, int] = (H, W)):
        self.paths = [Path(p) for p in paths]
        self.size = size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return to_tensor(read_bgr(self.paths[i]), *self.size)
