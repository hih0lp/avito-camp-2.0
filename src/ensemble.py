"""Признаки методов → калиброванная вероятность; стекинг и каскад.

Все признаки антисимметричные: при замене x на rot180(x) признак меняет знак.
Тогда логистическая регрессия без свободного члена даёт p(rot180(x)) = 1 - p(x),
то есть симметрия задачи сохраняется автоматически.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CLASSIFIERS = {"textline_x025", "doctr_crop"}
EPS = 1e-4


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def method_features(method: str, df: pd.DataFrame) -> pd.DataFrame:
    """Антисимметричные признаки одного метода. Знак «+» означает «скорее перевёрнут»."""
    # cnn_* — наша CNN (формат p_x / p_rot, как у классификаторов)
    if method in CLASSIFIERS or method.startswith("cnn"):
        return pd.DataFrame({f"{method}__dlogit": _logit(df.p_x) - _logit(df.p_rot)})
    # Распознаватель: если rot180(x) читается увереннее, чем x, то x перевёрнут
    return pd.DataFrame({
        f"{method}__ds": df.s_rot - df.s_x,
        f"{method}__dlogit": _logit(df.s_rot) - _logit(df.s_x),
        f"{method}__dlen": np.log1p(df.len_rot) - np.log1p(df.len_x),
    })


def load_features(features_dir: Path | list[Path], split: str, methods: list[str], ids: list[str]) -> pd.DataFrame:
    """features_dir — папка или список папок; файл метода ищется по порядку."""
    dirs = [features_dir] if isinstance(features_dir, (str, Path)) else list(features_dir)
    parts = []
    for spec in methods:
        # "rec_cyrillic_v5:lm" — тот же файл признаков + признак языковой модели по распознанному тексту
        m, _, opt = spec.partition(":")
        path = next(Path(d) / f"{split}_{m}.csv" for d in dirs if (Path(d) / f"{split}_{m}.csv").exists())
        raw = pd.read_csv(path, keep_default_na=False).set_index("image_id").loc[ids].reset_index(drop=True)
        parts.append(method_features(m, raw))
        if opt == "lm":
            parts.append(lm_features(m, raw))
    return pd.concat(parts, axis=1)


def lm_features(method: str, df: pd.DataFrame) -> pd.DataFrame:
    from .textlm import load_lm
    lm = load_lm()
    score_x = df.text_x.astype(str).map(lm.score)
    score_rot = df.text_rot.astype(str).map(lm.score)
    return pd.DataFrame({f"{method}__dlm": score_rot - score_x})


def make_model(C: float = 1.0):
    # with_mean=False: центрирование сломало бы антисимметрию
    return make_pipeline(StandardScaler(with_mean=False), LogisticRegression(C=C, fit_intercept=False, max_iter=1000))


def oof_predict(X: pd.DataFrame, y: np.ndarray, C: float = 1.0, seed: int = 42, n_splits: int = 5) -> np.ndarray:
    """Out-of-fold вероятности: каждая точка предсказана моделью, которая её не видела."""
    oof = np.zeros(len(y))
    for tr, va in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X, y):
        model = make_model(C).fit(X.iloc[tr], y[tr])
        oof[va] = model.predict_proba(X.iloc[va])[:, 1]
    return oof


def metrics(p: np.ndarray, y: np.ndarray) -> dict:
    pc = np.clip(p, EPS, 1 - EPS)
    return {
        "brier": float(np.mean((p - y) ** 2)),
        "1-brier": float(1 - np.mean((p - y) ** 2)),
        "acc": float(np.mean((p > 0.5) == y)),
        "logloss": float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))),
    }


def cascade(p_fast: np.ndarray, p_slow: np.ndarray, threshold: float) -> tuple[np.ndarray, float]:
    """Быстрая модель отвечает сама, если уверена (max(p, 1-p) >= threshold), иначе зовём медленную.

    Возвращает итоговые вероятности и долю кропов, ушедших в медленную модель.
    """
    confident = np.maximum(p_fast, 1 - p_fast) >= threshold
    return np.where(confident, p_fast, p_slow), float((~confident).mean())
