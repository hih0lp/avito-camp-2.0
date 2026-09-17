"""Методы определения ориентации. Каждый возвращает «сырые» признаки для x и rot180(x).

Классификаторы (textline_ori, docTR) дают p_x = P(180 | x) и p_rot = P(180 | rot180(x)).
Распознаватели (PaddleOCR rec, EasyOCR) дают уверенность чтения s_x и s_rot:
текст в правильной ориентации читается увереннее, поэтому признак — разность s_rot - s_x.
"""
import cv2
import numpy as np
import pandas as pd


def with_rotated(images: list[np.ndarray]) -> list[np.ndarray]:
    return images + [cv2.rotate(img, cv2.ROTATE_180) for img in images]


def split_pairs(values: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    return values[:n], values[n:]


class TextlineOri:
    """Готовый классификатор PaddleOCR (попытка 1)."""

    def __init__(self, model_name: str = "PP-LCNet_x0_25_textline_ori"):
        from paddleocr import TextLineOrientationClassification
        self.model = TextLineOrientationClassification(model_name=model_name, device="cpu")

    def __call__(self, images: list[np.ndarray]) -> pd.DataFrame:
        probs = []
        for res in self.model.predict(with_rotated(images), batch_size=64):
            # class_ids в paddlex общий на батч, поэтому берём label_names
            score = float(res["scores"][0])
            probs.append(score if res["label_names"][0] == "180_degree" else 1.0 - score)
        p_x, p_rot = split_pairs(np.asarray(probs), len(images))
        return pd.DataFrame({"p_x": p_x, "p_rot": p_rot})


class DoctrCropOrientation:
    """docTR mobilenet_v3_small_crop_orientation: 4 класса [0, -90, 180, 90]."""

    def __init__(self, device: str = "cpu"):
        import torch
        from doctr.models import crop_orientation_predictor
        self.torch = torch
        self.predictor = crop_orientation_predictor(pretrained=True)
        self.predictor.model.to(device)
        self.device = device
        classes = list(self.predictor.model.cfg["classes"])
        self.i0, self.i180 = classes.index(0), classes.index(180)

    def __call__(self, images: list[np.ndarray]) -> pd.DataFrame:
        rgb = [img[:, :, ::-1].copy() for img in with_rotated(images)]
        probs = []
        # Стандартный predictor отдаёт только top-1, поэтому повторяем его шаги и берём полный softmax
        with self.torch.inference_mode():
            for batch in self.predictor.pre_processor(rgb):
                logits = self.predictor.model(batch.to(self.device)).float()
                sm = self.torch.softmax(logits, dim=1).cpu().numpy()
                # Модель знает ещё про 90/270; нас интересует только выбор между 0 и 180
                probs.append(sm[:, self.i180] / (sm[:, self.i0] + sm[:, self.i180] + 1e-12))
        p_x, p_rot = split_pairs(np.concatenate(probs), len(images))
        return pd.DataFrame({"p_x": p_x, "p_rot": p_rot})


class PaddleRec:
    """Уверенность распознавателя PaddleOCR (eslav_* — русский/укр/бел, cyrillic_* — вся кириллица)."""

    def __init__(self, model_name: str):
        from paddleocr import TextRecognition
        self.model = TextRecognition(model_name=model_name, device="cpu")

    def __call__(self, images: list[np.ndarray]) -> pd.DataFrame:
        scores, texts = [], []
        for res in self.model.predict(with_rotated(images), batch_size=64):
            scores.append(float(res["rec_score"]))
            texts.append(res["rec_text"].strip())
        n = len(images)
        s_x, s_rot = split_pairs(np.asarray(scores), n)
        # сам текст сохраняем для языковой модели (src/textlm.py)
        return pd.DataFrame({"s_x": s_x, "s_rot": s_rot,
                             "len_x": [len(t) for t in texts[:n]], "len_rot": [len(t) for t in texts[n:]],
                             "text_x": texts[:n], "text_rot": texts[n:]})


class EasyOCRRec:
    """Уверенность распознавателя EasyOCR (ru + en). Кроп целиком подаётся как одна строка."""

    def __init__(self, device: str = "cpu"):
        import easyocr
        self.reader = easyocr.Reader(["ru", "en"], gpu=device != "cpu", verbose=False)

    def _read(self, img: np.ndarray) -> tuple[float, int]:
        h, w = img.shape[:2]
        # detail=1: [(bbox, text, conf)]; бокс = весь кроп, детектор EasyOCR не нужен
        out = self.reader.recognize(img, horizontal_list=[[0, w, 0, h]], free_list=[], detail=1)
        if not out:
            return 0.0, 0
        _, text, conf = out[0]
        return float(conf), len(text.strip())

    def __call__(self, images: list[np.ndarray]) -> pd.DataFrame:
        results = [self._read(img) for img in with_rotated(images)]
        conf = np.array([r[0] for r in results])
        length = np.array([r[1] for r in results])
        s_x, s_rot = split_pairs(conf, len(images))
        len_x, len_rot = split_pairs(length, len(images))
        return pd.DataFrame({"s_x": s_x, "s_rot": s_rot, "len_x": len_x, "len_rot": len_rot})


def build_method(name: str, device: str = "cpu"):
    factories = {
        "textline_x025": lambda: TextlineOri("PP-LCNet_x0_25_textline_ori"),
        "doctr_crop": lambda: DoctrCropOrientation(device),
        "rec_eslav_v5": lambda: PaddleRec("eslav_PP-OCRv5_mobile_rec"),
        "rec_cyrillic_v5": lambda: PaddleRec("cyrillic_PP-OCRv5_mobile_rec"),
        "rec_cyrillic_v3": lambda: PaddleRec("cyrillic_PP-OCRv3_mobile_rec"),
        "rec_en_v5": lambda: PaddleRec("en_PP-OCRv5_mobile_rec"),
        "easyocr_ru": lambda: EasyOCRRec(device),
    }
    return factories[name]()


METHODS = ["textline_x025", "doctr_crop", "rec_eslav_v5", "rec_cyrillic_v5", "rec_cyrillic_v3", "easyocr_ru", "rec_en_v5"]
