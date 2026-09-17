"""Генерирует синтетическую размеченную валидацию в data/synth_val (3000 кропов, seed=42).

    python make_synth_val.py
"""
import argparse
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1] # корень проекта
sys.path.insert(0, str(PROJECT))

from src.synth import make_synthetic_set # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=PROJECT / "data" / "synth_val")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = make_synthetic_set(args.out, args.n, seed=args.seed)
    print(f"Сохранено {len(df)} кропов в {args.out}, доля перевёрнутых {df.label.mean():.3f}")


if __name__ == "__main__":
    main()
