"""Draw every figure from results/summary.json (no GPU).

    python analysis/figures.py
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import DEPTHS, RECIPE, RESULTS, ROOT, STEPS, TOKENIZERS, WIDTHS

FIG = ROOT / "figures"
S = json.loads((RESULTS / "summary.json").read_text())
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948", "#4d4d4d"]
TOK_LABEL = {"char": "char (65)", "byte": "byte (256)", "word_2048": "word 2048", "word_8192": "word 8192",
             "sp_bpe_512": "BPE 512", "sp_bpe_2048": "BPE 2048", "sp_unigram_2048": "unigram 2048",
             "gpt2_bpe": "GPT-2 BPE (50,257)"}
RECIPE_LABEL = {"gpt2_textbook": "textbook", "plus_rmsnorm_rope": "+ RMSNorm, RoPE", "plus_wsd": "+ WSD",
                "plus_muon": "+ Muon", "muon_cosine": "Muon, cosine"}
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e4e4e4", "legend.frameon": False, "lines.linewidth": 1.5})


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] figures/{name}.png")


def band(ax, c, color, label, x="tokens", scale=1e6, ls="-"):
    xs, m, s = np.array(c[x]) / scale, np.array(c["mean"]), np.array(c["std"])
    ax.plot(xs, m, color=color, label=label, ls=ls)
    ax.fill_between(xs, m - s, m + s, color=color, alpha=0.18, lw=0)


def tokenizers():
    T, F = S["tokenizers"]["by_tokenizer"], S["tokenizers"]["fix"]
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.8), gridspec_kw={"width_ratios": [1.25, 0.95, 1]})
    ax = axes[0]
    for i, n in enumerate(TOKENIZERS):
        band(ax, T[n]["curve"], PAL[i], TOK_LABEL[n], x="steps", scale=1)
    ax.set_ylim(2.0, 3.3)
    ax.set_yticks(np.arange(2.0, 3.01, 0.2))
    ax.set_xlabel("optimizer step (8,192 tokens each)")
    ax.set_ylabel("validation bits per byte")
    ax.legend(fontsize=5.4, ncol=2, loc="upper right", handlelength=1.2, columnspacing=0.8)
    ax.set_title("(a) validation bpb", loc="left", fontsize=8.5)
    ax = axes[1]
    order = sorted(TOKENIZERS, key=lambda n: T[n]["final_bpb"]["mean"])
    for i, n in enumerate(order):
        f, b = T[n]["final_bpb"], T[n]["best_bpb"]
        ax.errorbar(f["mean"], i + 0.15, xerr=f["std"], fmt="o", color=PAL[TOKENIZERS.index(n)], ms=4, capsize=2,
                    label="final" if i == 0 else None)
        ax.errorbar(b["mean"], i - 0.15, xerr=b["std"], fmt="s", color=PAL[TOKENIZERS.index(n)], mfc="white", ms=4,
                    capsize=2, label="best" if i == 0 else None)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([TOK_LABEL[n] for n in order], fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlabel("validation bits per byte")
    ax.legend(fontsize=6.5, loc="center right")
    ax.set_title("(b) final and best", loc="left", fontsize=8.5)
    ax = axes[2]
    band(ax, T["gpt2_bpe"]["curve"], PAL[7], "GPT-2 BPE", x="steps", scale=1)
    band(ax, F["gpt2_bpe_500steps"]["curve"], PAL[0], "GPT-2, 500 steps", x="steps", scale=1)
    band(ax, F["gpt2_bpe_dropout0.3"]["curve"], PAL[2], "GPT-2, dropout 0.3", x="steps", scale=1)
    band(ax, T["char"]["curve"], PAL[8], "char", x="steps", scale=1, ls="--")
    ax.set_ylim(2.0, 3.3)
    ax.set_yticks(np.arange(2.0, 3.01, 0.2))
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("validation bits per byte")
    ax.legend(fontsize=6.3, loc="upper left")
    ax.set_title("(c) fixes for GPT-2 BPE", loc="left", fontsize=8.5)
    fig.tight_layout()
    save(fig, "tokenizers")


def recipe():
    R = S["recipe"]
    Q = R["positions"]["configs"]
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.8), gridspec_kw={"width_ratios": [1.15, 1, 1]})
    ax = axes[0]
    for i, n in enumerate(RECIPE):
        band(ax, R["configs"][n]["curve"], PAL[i], RECIPE_LABEL[n], ls="--" if n == "muon_cosine" else "-")
    ax.set_ylim(1.75, 3.0)
    ax.set_xlabel("training tokens (millions)")
    ax.set_ylabel("validation bpb (first 524,288 tokens)")
    ax.legend(fontsize=5.6, loc="upper right")
    ax.set_title("(a) the recipe, 5 seeds", loc="left", fontsize=8.5)
    ax = axes[1]
    rows = [("RoPE alone", R["parts"]["rope_alone"]), ("RMSNorm alone", R["parts"]["rmsnorm_alone"]),
            ("+ RMSNorm, RoPE", R["chain"]["gpt2_textbook->plus_rmsnorm_rope"]),
            ("+ WSD (AdamW)", R["chain"]["plus_rmsnorm_rope->plus_wsd"]),
            ("+ Muon (WSD)", R["chain"]["plus_wsd->plus_muon"]),
            ("WSD, under Muon", R["factorial"]["schedule_under_muon"]),
            ("Muon, under cosine", R["factorial"]["optimizer_under_cosine"]),
            ("Muon, vs matched AdamW", R["parts"]["optimizer_matched"])]
    colors = [PAL[8], PAL[8], PAL[1], PAL[2], PAL[3], PAL[8], PAL[8], PAL[8]]
    for i, (lab, d) in enumerate(rows):
        ax.errorbar(d["mean_diff"], i, xerr=d["std_diff"], fmt="o", color=colors[i], ms=4, capsize=2,
                    mfc="white" if colors[i] == PAL[8] else colors[i])
    ax.axvline(0, color="#999999", lw=0.8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlabel("paired change in final bpb")
    ax.set_title("(b) each change (− = better)", loc="left", fontsize=8.5)
    ax = axes[2]
    band(ax, R["configs"]["gpt2_textbook"]["curve"], PAL[0], "learned, 16.4M")
    band(ax, Q["textbook_steps4000"]["curve"], PAL[0], "learned, 65.5M", ls="--")
    band(ax, R["configs"]["textbook_rope"]["curve"], PAL[1], "RoPE, 16.4M")
    band(ax, Q["textbook_rope_steps4000"]["curve"], PAL[1], "RoPE, 65.5M", ls="--")
    c = Q["textbook_nope"]["curve"]
    ax.plot(np.array(c["tokens"]) / 1e6, c["mean"], color=PAL[8], ls=":", marker="x", ms=4,
            label="none, 16.4M")
    ax.set_ylim(1.75, 3.45)
    ax.set_yticks(np.arange(1.8, 2.81, 0.2))
    ax.set_xlabel("training tokens (millions)")
    ax.set_ylabel("validation bpb")
    ax.legend(fontsize=6, loc="upper right", title="positions, tokens", title_fontsize=6)
    ax.set_title("(c) position encoding, textbook", loc="left", fontsize=8.5)
    fig.tight_layout()
    save(fig, "recipe")


def scaling():
    P, Dp = S["scaling"]["points"], S["scaling"]["depth_points"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    for j, st in enumerate(STEPS):
        pts = [P[f"{w}_{st}"] for w in WIDTHS]
        dps = [Dp[f"{L}_{st}"] for L in DEPTHS]
        tok = pts[0]["tokens"] / 1e6
        for ax, key in ((axes[0], "params_non_embedding"), (axes[1], "flops_6ND")):
            ax.errorbar([p[key] for p in pts], [p["mean"] for p in pts], yerr=[p["std"] for p in pts], marker="o",
                        ms=4, capsize=2, color=PAL[j], label=f"{tok:.1f}M tokens, width 128 to 512 (4 layers)")
            ax.errorbar([p[key] for p in dps], [p["mean"] for p in dps], yerr=[p["std"] for p in dps], marker="s",
                        ms=4, mfc="white", capsize=2, color=PAL[j], ls=":",
                        label=f"{tok:.1f}M tokens, 2 to 8 layers (width 256)")
    axes[0].set_xscale("log")
    axes[1].set_xscale("log")
    axes[0].set_xlabel("non-embedding parameters")
    axes[1].set_xlabel("training FLOPs (6ND)")
    axes[0].set_ylabel("validation bpb (first 2,097,152 tokens)")
    axes[1].set_ylabel("validation bpb (first 2,097,152 tokens)")
    axes[1].yaxis.set_tick_params(labelleft=True)
    axes[1].legend(fontsize=5.8, loc="upper right")
    axes[0].set_title("(a) bpb against model size", loc="left", fontsize=8.5)
    axes[1].set_title("(b) bpb against compute", loc="left", fontsize=8.5)
    for w, p in ((w, P[f"{w}_1000"]) for w in WIDTHS):
        axes[0].annotate(f"d={w}", (p["params_non_embedding"], p["mean"]), textcoords="offset points", xytext=(3, 4),
                         fontsize=6.5, color="#555555")
    fig.tight_layout()
    save(fig, "scaling")


def swiglu():
    W = S["swiglu"]["configs"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    ax = axes[0]
    band(ax, W["gelu"]["curve"], PAL[0], "GELU, lr 1e-3")
    band(ax, W["swiglu"]["curve"], PAL[7], "SwiGLU, lr 1e-3")
    band(ax, W["gelu_steps6000"]["curve"], PAL[0], "GELU, 2× tokens", ls="--")
    band(ax, W["swiglu_steps6000"]["curve"], PAL[7], "SwiGLU, 2× tokens", ls="--")
    ax.set_ylim(1.6, 2.1)
    ax.set_xlabel("training tokens (millions)")
    ax.set_ylabel("validation bpb (first 524,288 tokens)")
    ax.legend(fontsize=6.5)
    ax.set_title("(a) curves, mean of 5 seeds", loc="left", fontsize=8.5)
    ax = axes[1]
    cols = [("gelu_lr5e-4", "swiglu_lr5e-4"), ("gelu", "swiglu"), ("gelu_lr2e-3", "swiglu_lr2e-3"),
            ("gelu_lr4e-3", "swiglu_lr4e-3"), ("gelu_steps6000", "swiglu_steps6000")]
    for x, (g, s) in enumerate(cols):
        for name, col, off in ((g, PAL[0], -0.12), (s, PAL[7], 0.12)):
            ax.errorbar(x + off, W[name]["mean"], yerr=W[name]["std"], fmt="o", color=col, ms=4, capsize=2)
            ax.scatter([x + off] * len(W[name]["values"]), W[name]["values"], s=8, color=col, alpha=0.5)
    ax.axvline(3.5, color="#bbbbbb", lw=0.8)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(["lr 5e-4", "lr 1e-3", "lr 2e-3", "lr 4e-3", "lr 1e-3,\n2× tokens"], fontsize=7)
    ax.set_ylabel("final validation bpb (first 4,194,304 tokens)")
    ax.plot([], [], "o", color=PAL[0], label="GELU")
    ax.plot([], [], "o", color=PAL[7], label="SwiGLU")
    ax.legend(fontsize=7)
    ax.set_title("(b) final bpb, mean ± std, dots = seeds", loc="left", fontsize=8.5)
    fig.tight_layout()
    save(fig, "swiglu")


if __name__ == "__main__":
    tokenizers()
    recipe()
    scaling()
    swiglu()
