"""Синтетическая размеченная валидация: русский/латинский текст, отрисованный системными шрифтами.

Нужна, чтобы сравнивать методы по настоящему Brier: на тесте меток нет,
а у методов «сравни уверенность распознавателя на x и rot180(x)» самосогласованность всегда 1.
Домен отличается от теста (фото Авито), поэтому смотрим на относительный порядок методов,
а не на абсолютные числа.
"""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from fontTools.ttLib import TTCollection, TTFont
from PIL import Image, ImageDraw, ImageFont

# Системные шрифты: macOS, Linux, Windows. Набор шрифтов зависит от ОС, поэтому синтетика, сгенерированная
# на другой системе, будет немного другой — для точного воспроизведения используйте готовую папку data/synth_val.
FONT_DIRS = [ # (папка, искать ли в подпапках)
    (Path("/System/Library/Fonts"), False), (Path("/System/Library/Fonts/Supplemental"), False), # macOS
    (Path("/Library/Fonts"), False),
    (Path("/usr/share/fonts"), True), (Path("/usr/local/share/fonts"), True), # Linux
    (Path.home() / ".fonts", True),
    (Path("C:/Windows/Fonts"), False), # Windows
]
# Шрифты, которые рисуют не буквы, а значки/орнаменты
FONT_BLOCKLIST = ("Ornaments", "Webdings", "Wingdings", "Symbol", "Emoji", "LastResort", "Zapf", "Bodoni Ornaments")

WORDS = """
продам куплю новый новая б/у торг обмен доставка самовывоз гарантия оригинал размер цена скидка
телефон диван кровать шкаф стол стулья холодильник стиральная машина велосипед коляска куртка
пальто кроссовки платье сумка ноутбук смартфон наушники зарядка аккумулятор шины диски запчасти
разбор авто ремонт квартира комната аренда посуточно студия этаж площадь кухня ванная балкон
магазин склад работа вакансия курьер водитель продавец зарплата график опыт звоните пишите
молоко сливочное масло хлеб сыр шоколад кофе чай конфеты печенье орехи мёд вода сок
изготовитель производитель состав срок годности хранить при температуре масса нетто
Москва Санкт-Петербург Новосибирск Екатеринбург Казань Омск Самара Ростов Краснодар
детский женская мужская натуральная кожа хлопок шерсть зимняя летняя весна осень
инструкция внимание осторожно выход вход открыто закрыто акция хит продаж качество
""".split()
LATIN = """Samsung Apple iPhone Xiaomi Bosch LG Sony Nike Adidas Toyota BMW Lada IKEA Philips
Zara Kinder Nestle Carte Noire Original Size Made in China Premium Quality SALE NEW Pro Max""".split()


def _has_cyrillic(path: Path) -> list[int]:
    """Индексы шрифтов в файле, где есть кириллица (для .ttc их несколько)."""
    try:
        fonts = TTCollection(str(path)).fonts if path.suffix.lower() == ".ttc" else [TTFont(str(path), lazy=True)]
    except Exception:
        return []
    good = []
    for idx, font in enumerate(fonts):
        try:
            cmap = font.getBestCmap() or {}
        except Exception:
            continue
        if all(ord(ch) in cmap for ch in "АБЖЫЯабжыя0123456789"):
            good.append(idx)
    return good


def find_cyrillic_fonts() -> list[tuple[str, int]]:
    fonts = []
    for d, recursive in FONT_DIRS:
        if not d.exists():
            continue
        # на Linux шрифты разложены по подпапкам, поэтому там ищем рекурсивно
        for path in sorted(d.rglob("*") if recursive else d.glob("*")):
            if path.suffix.lower() not in (".ttf", ".otf", ".ttc") or any(b in path.name for b in FONT_BLOCKLIST):
                continue
            # из больших коллекций берём не больше 3 начертаний, чтобы не перекосить выборку
            fonts += [(str(path), i) for i in _has_cyrillic(path)[:3]]
    return fonts


def random_text(rng: np.random.Generator) -> str:
    kind = rng.random()
    if kind < 0.55:
        words = rng.choice(WORDS, size=rng.integers(1, 4))
        text = " ".join(words)
        if rng.random() < 0.3:
            text = text.upper()
        elif rng.random() < 0.3:
            text = text.capitalize()
        return text
    if kind < 0.7:
        return " ".join(rng.choice(LATIN, size=rng.integers(1, 3)))
    if kind < 0.8:
        return f"{rng.integers(1, 999)} {rng.integers(0, 999):03d} р." if rng.random() < 0.5 else f"{rng.integers(10, 99999)} руб."
    if kind < 0.9:
        return f"+7 ({rng.integers(900, 999)}) {rng.integers(100, 999)}-{rng.integers(10, 99)}-{rng.integers(10, 99)}"
    return f"{rng.choice(WORDS)} {rng.integers(1, 500)}{rng.choice(['', ' шт', ' г', ' мл', ' см', '%'])}"


def _background(h: int, w: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Фон (градиент + шум) и контрастный к нему цвет текста."""
    c1, c2 = rng.integers(0, 256, 3), rng.integers(0, 256, 3)
    t = np.linspace(0, 1, w)[None, :, None]
    bg = (c1 * (1 - t) + c2 * t) * np.ones((h, 1, 1))
    bg += rng.normal(0, rng.uniform(0, 12), bg.shape)
    lum = bg.mean()
    text_color = rng.integers(0, 70, 3) if lum > 128 else rng.integers(185, 256, 3)
    return np.clip(bg, 0, 255).astype(np.uint8), text_color


def render_crop(text: str, font: tuple[str, int], rng: np.random.Generator) -> np.ndarray:
    """Кроп «как от детектора»: текст, неровные поля, лёгкий наклон, blur, шум, JPEG. Возвращает BGR."""
    size = int(rng.integers(28, 72))
    pil_font = ImageFont.truetype(font[0], size=size, index=font[1])
    left, top, right, bottom = pil_font.getbbox(text)
    tw, th = right - left, bottom - top
    pad_x, pad_y = int(rng.uniform(0.1, 0.8) * size), int(rng.uniform(0.2, 0.8) * size)
    w, h = tw + 2 * pad_x, th + 2 * pad_y

    bg, color = _background(h, w, rng)
    img = Image.fromarray(bg)
    ImageDraw.Draw(img).text((pad_x - left, pad_y - top), text, font=pil_font, fill=tuple(int(c) for c in color))
    arr = np.array(img)[:, :, ::-1] # RGB -> BGR

    # Лёгкий поворот ±4°: детектор выдаёт не идеально выровненные боксы
    angle = rng.uniform(-4, 4)
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    arr = cv2.warpAffine(arr, m, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # Обрезаем поля неравномерно, иногда задевая буквы (как у неточного бокса)
    cut = lambda pad: int(rng.uniform(-0.05, 0.9) * pad)
    y0, y1 = max(0, cut(pad_y)), h - max(0, cut(pad_y))
    x0, x1 = max(0, cut(pad_x)), w - max(0, cut(pad_x))
    arr = arr[y0:y1, x0:x1]

    # Итоговая высота как в тесте (медиана ~42 px), с даунскейлом — отсюда пикселизация мелкого текста
    target_h = int(np.clip(rng.lognormal(np.log(40), 0.5), 12, 160))
    scale = target_h / arr.shape[0]
    arr = cv2.resize(arr, (max(8, int(arr.shape[1] * scale)), target_h), interpolation=cv2.INTER_AREA)

    if rng.random() < 0.5:
        k = int(rng.choice([3, 5]))
        arr = cv2.GaussianBlur(arr, (k, k), 0)
    if rng.random() < 0.7:
        quality = int(rng.integers(25, 95))
        arr = cv2.imdecode(cv2.imencode(".jpg", arr, [cv2.IMWRITE_JPEG_QUALITY, quality])[1], cv2.IMREAD_COLOR)
    return arr


def make_synthetic_set(out_dir: Path, n: int, seed: int = 42) -> pd.DataFrame:
    """Генерирует n кропов; каждый с вероятностью 0.5 переворачивается на 180° (label=1)."""
    rng = np.random.default_rng(seed)
    fonts = find_cyrillic_fonts()
    if not fonts:
        raise RuntimeError("Не найдено ни одного шрифта с кириллицей (нужны, например, DejaVu или Liberation)")
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i in range(n):
        text = random_text(rng)
        font = fonts[int(rng.integers(len(fonts)))]
        img = render_crop(text, font, rng)
        label = int(rng.random() < 0.5)
        if label:
            img = cv2.rotate(img, cv2.ROTATE_180)
        image_id = f"val_{i:05d}"
        cv2.imwrite(str(img_dir / f"{image_id}.png"), img)
        rows.append({"image_id": image_id, "label": label, "text": text, "font": Path(font[0]).name})

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "labels.csv", index=False)
    return df
