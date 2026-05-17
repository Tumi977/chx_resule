"""Sanity-check the local evaluation pipeline.

Three checks on a small validation subset:
    1) pred = noisy        -> score should be 0 (no improvement over baseline)
    2) pred = clean        -> score should be ~100 (essentially perfect)
    3) pred = noisy + extra small noise -> score should be 0 (worse than noisy)

If these match expectation, our local_eval is consistent with the official one.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from datasets.pc_dataset import ValidateConfig, make_validation_set
from eval.local_eval import evaluate_one, aggregate, fmt_summary


def main():
    val_set = make_validation_set(ValidateConfig(noise_sigma=0.010))[:5]
    print(f"running sanity on {len(val_set)} shapes")

    print("\n--- (1) pred = noisy ---")
    rs = []
    for rel, n, c, _ in val_set:
        rs.append(evaluate_one(rel, n, c, n))
    print(fmt_summary(aggregate(rs)))

    print("\n--- (2) pred = clean ---")
    rs = []
    for rel, n, c, _ in val_set:
        rs.append(evaluate_one(rel, n, c, c))
    print(fmt_summary(aggregate(rs)))

    print("\n--- (3) pred = noisy + extra noise (worse) ---")
    rs = []
    for rel, n, c, _ in val_set:
        rng = np.random.default_rng(7)
        worse = (n + rng.normal(0, 0.01, size=n.shape)).astype(np.float32)
        rs.append(evaluate_one(rel, n, c, worse))
    print(fmt_summary(aggregate(rs)))


if __name__ == "__main__":
    main()
