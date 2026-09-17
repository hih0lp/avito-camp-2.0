"""Символьная n-граммная языковая модель: насколько распознанная строка похожа на настоящий текст.

Если прочитать перевёрнутый кроп, обычно получается бессмыслица («SOXH», «bKaabb»),
поэтому признак lm(текст rot180(x)) − lm(текст x) помогает там, где уверенность распознавателя
в обеих ориентациях похожа.

Обучение — на строках из внешних датасетов (data/packed/*.meta.csv): печатная кириллица (Printed-6/10)
и реальные английские слова (Union14M train). Тестовые данные не используются.
"""
import math
import pickle
from collections import defaultdict
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
CORPORA = ["printed6", "printed10", "union_train"]
CACHE = PROJECT / "data/textlm_4gram.pkl"
BOS, EOS = "\x02", "\x03"


class CharNgramLM:
    def __init__(self, order: int = 4, alpha: float = 2.0):
        self.order = order
        self.alpha = alpha # сила сглаживания: сколько «псевдо-наблюдений» из модели меньшего порядка
        self.counts = [defaultdict(lambda: defaultdict(int)) for _ in range(order)]
        self.totals = [defaultdict(int) for _ in range(order)]
        self.vocab = set()

    def fit(self, texts) -> "CharNgramLM":
        for text in texts:
            s = BOS * (self.order - 1) + text.lower() + EOS
            self.vocab.update(s)
            for i in range(self.order - 1, len(s)):
                for k in range(self.order): # k — длина контекста
                    ctx = s[i - k:i]
                    self.counts[k][ctx][s[i]] += 1
                    self.totals[k][ctx] += 1
        # defaultdict с lambda не сериализуется — переводим в обычные словари
        self.counts = [{c: dict(d) for c, d in level.items()} for level in self.counts]
        self.totals = [dict(level) for level in self.totals]
        return self

    def prob(self, ctx: str, ch: str) -> float:
        """Рекурсивное сглаживание: P_k = (N(ctx_k, ch) + α·P_{k-1}) / (N(ctx_k) + α)."""
        p = 1.0 / (len(self.vocab) + 1)
        for k in range(self.order):
            c = ctx[len(ctx) - k:] if k else ""
            n_ctx = self.totals[k].get(c, 0)
            n = self.counts[k].get(c, {}).get(ch, 0)
            p = (n + self.alpha * p) / (n_ctx + self.alpha)
        return p

    def score(self, text: str) -> float:
        """Средний log P на символ (включая конец строки). Пустая строка — очень плохой текст."""
        if not isinstance(text, str) or not text.strip():
            return -8.0
        s = BOS * (self.order - 1) + text.lower() + EOS
        lp = [math.log(self.prob(s[i - self.order + 1:i], s[i])) for i in range(self.order - 1, len(s))]
        return sum(lp) / len(lp)


_LM = None


def load_lm() -> CharNgramLM:
    """Модель обучается один раз и кэшируется на диск (только данные, без класса — pickle не зависит от путей импорта)."""
    global _LM
    if _LM is not None:
        return _LM
    lm = CharNgramLM()
    if CACHE.exists():
        with open(CACHE, "rb") as f:
            state = pickle.load(f)
        lm.counts, lm.totals, lm.vocab = state["counts"], state["totals"], state["vocab"]
    else:
        texts = []
        for name in CORPORA:
            meta = pd.read_csv(PROJECT / f"data/packed/{name}.meta.csv", usecols=["text"])
            texts += meta.text.dropna().astype(str).tolist()
        lm.fit(texts)
        with open(CACHE, "wb") as f:
            pickle.dump({"counts": lm.counts, "totals": lm.totals, "vocab": lm.vocab}, f)
    _LM = lm
    return lm
