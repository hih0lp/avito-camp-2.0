"""Чтение данных: список id из sample_submission и картинки по id."""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def load_ids(sample_path: Path) -> list[str]:
    return pd.read_csv(sample_path)["image_id"].astype(str).tolist()


def ids_from_dir(images_dir: Path) -> list[str]:
    """Имена файлов (без расширения) как image_id — если файла-примера нет."""
    return sorted(p.stem for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTS)


def resolve_ids(images_dir: Path, sample_path: Path) -> list[str]:
    """Порядок из sample_submission.csv, если он есть; иначе просто все картинки из папки."""
    return load_ids(sample_path) if sample_path.exists() else ids_from_dir(images_dir)


def resolve_path(images_dir: Path, image_id: str) -> Path:
    """image_id в sample_submission без расширения (test_00000), ищем подходящий файл."""
    path = images_dir / image_id
    if path.exists():
        return path
    for ext in IMG_EXTS:
        candidate = images_dir / f"{image_id}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Нет файла для image_id={image_id}")


def read_image(path: Path) -> np.ndarray:
    # cv2.imread ломается на не-ASCII путях, поэтому декодируем байты сами.
    # Результат в BGR — именно этот формат ожидает PaddleOCR.
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Не удалось прочитать {path}")
    return img
