"""Shared helpers: locate runs and read their records (no GPU, no torch).

Every script runs from the repository root.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "sweeps"
RESULTS = ROOT / "results"
TOKENIZERS = ["char", "byte", "word_2048", "word_8192", "sp_bpe_512", "sp_bpe_2048", "sp_unigram_2048", "gpt2_bpe"]
TOK_SEEDS = [1337, 42, 1]
RECIPE = ["gpt2_textbook", "plus_rmsnorm_rope", "plus_wsd", "plus_muon", "muon_cosine"]
RECIPE_SEEDS = [1337, 42, 1, 2, 3]
RECIPE_PARTS = ["textbook_rope", "textbook_rmsnorm", "adamw_matched"]
WIDTHS = [128, 176, 256, 352, 512]
STEPS = [1000, 2000, 4000]
DEPTHS = [2, 4, 6, 8]  # the depth series at width 256
PAIR_SEEDS = [1337, 42]
SWIGLU_SEEDS = [1337, 42, 1, 2, 3]


def run_dir(sweep: str, name: str, seed: int) -> Path:
    return RUNS / sweep / f"{name}_s{seed}"


def metrics(sweep: str, name: str, seed: int) -> dict:
    return json.loads((run_dir(sweep, name, seed) / "metrics.json").read_text())


def eval_log(sweep: str, name: str, seed: int) -> list[dict]:
    return [json.loads(l) for l in (run_dir(sweep, name, seed) / "eval_log.jsonl").read_text().splitlines()]


def passes(sweep: str, name: str, seed: int) -> float:
    """Passes over the training split, as printed by nanolm's trainer."""
    log = (run_dir(sweep, name, seed) / "stdout.log").read_text()
    return float(re.search(r"\(([\d.]+) passes over train\)", log).group(1))


def scaling_run(width: int, steps: int) -> tuple[str, str]:
    """(sweep, name) of a scaling point; the 256-wide, 1000-step point is recipe/plus_muon."""
    return ("recipe", "plus_muon") if (width, steps) == (256, 1000) else ("scaling", f"d{width}_steps{steps}")


def describe(x) -> dict:
    x = np.asarray(x, dtype=float)
    return {"mean": float(x.mean()), "std": float(x.std(ddof=1)) if len(x) > 1 else 0.0, "values": x.tolist()}


def paired(a, b) -> dict:
    """Difference b - a over matched seeds, with nanolm.analyze's paired t-test."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    d = b - a
    p = float(stats.ttest_rel(b, a).pvalue) if len(d) > 1 and d.std() > 0 else float("nan")
    return {"mean_diff": float(d.mean()), "std_diff": float(d.std(ddof=1)) if len(d) > 1 else 0.0, "p": p,
            "n": len(d), "all_same_sign": bool((d > 0).all() or (d < 0).all())}


def noise_floor(groups) -> float:
    """nanolm.analyze's seed noise floor: median within-configuration std."""
    return float(np.median([np.std(g, ddof=1) for g in groups]))


def save(name: str, obj) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{name}.json").write_text(json.dumps(obj, indent=1))
    print(f"[saved] results/{name}.json")
