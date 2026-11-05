"""Collect every summary number (means, paired tests, curves) from the run records into
results/summary.json (no GPU).

Statistics follow nanolm.analyze: mean and sample std over seeds, and a paired t-test on the
per-seed differences (the same seeds in both arms).

    python analysis/summarize.py
"""
from __future__ import annotations

import json
import math

import numpy as np

from common import (DEPTHS, PAIR_SEEDS, ROOT, RECIPE, RECIPE_PARTS, RECIPE_SEEDS, STEPS, SWIGLU_SEEDS, TOK_SEEDS, TOKENIZERS, WIDTHS, describe, eval_log,
                    metrics, noise_floor, paired, passes, save, scaling_run)

# nanolm's tutorial, section 1.8 (one seed, 1337): final and best validation bpb, best step
PUBLISHED_TOKENIZERS = {"byte": (2.092, 2.088, 1750), "char": (2.105, 2.105, 1750), "word_2048": (2.080, 2.047, 1250),
                        "sp_bpe_512": (2.163, 2.123, 1250), "sp_bpe_2048": (2.368, 2.104, 500),
                        "sp_unigram_2048": (2.340, 2.113, 500), "word_8192": (2.318, 2.113, 750),
                        "gpt2_bpe": (2.825, 2.171, 500)}


def curve(sweep, name, seeds):
    """Mean and std over seeds of the periodic validation bpb, by training tokens."""
    logs = [eval_log(sweep, name, s) for s in seeds]
    tok = [e["train_tokens"] for e in logs[0]]
    a = np.array([[e["bpb"] for e in lg] for lg in logs])
    return {"tokens": tok, "steps": [e["step"] for e in logs[0]], "mean": a.mean(0).tolist(),
            "std": a.std(0, ddof=1).tolist(), "per_seed": a.tolist()}


def tokenizers():
    out = {}
    for n in TOKENIZERS:
        ms = [metrics("tokenizers", n, s) for s in TOK_SEEDS]
        logs = [eval_log("tokenizers", n, s) for s in TOK_SEEDS]
        best = [min(lg, key=lambda e: e["bpb"]) for lg in logs]
        out[n] = {"tokenizer": ms[0]["tokenizer"], "params": ms[0]["params"],
                  "params_non_embedding": ms[0]["params_non_embedding"],
                  "vocab_rows": (ms[0]["params"] - ms[0]["params_non_embedding"]) // 256,
                  "bytes_per_token": ms[0]["val_bytes_per_token"], "val_tokens": ms[0]["val_tokens"],
                  "val_bytes": ms[0]["val_bytes"],
                  "periodic_eval_tokens": logs[0][-1]["tokens"], "passes": passes("tokenizers", n, TOK_SEEDS[0]),
                  "final_bpb": describe([m["val_bpb"] for m in ms]),
                  "final_loss": describe([m["val_loss"] for m in ms]),
                  "final_ppl": describe([m["val_ppl"] for m in ms]),
                  "best_bpb": describe([b["bpb"] for b in best]), "best_step": [b["step"] for b in best],
                  "overfit": describe([m["val_bpb"] - b["bpb"] for m, b in zip(ms, best)]),
                  "published_final_best_step": PUBLISHED_TOKENIZERS[n],
                  "curve": curve("tokenizers", n, TOK_SEEDS)}
    final = {n: np.array(out[n]["final_bpb"]["values"]) for n in TOKENIZERS}
    res = {"by_tokenizer": out, "noise_floor": noise_floor(final.values()),
           "vs_char_final": {n: paired(final["char"], final[n]) for n in TOKENIZERS if n != "char"},
           "word_2048_vs_byte_final": paired(final["byte"], final["word_2048"])}
    gpt2 = final["gpt2_bpe"]
    best_gpt2 = np.array(out["gpt2_bpe"]["best_bpb"]["values"])
    fix = {}
    for n in ("gpt2_bpe_500steps", "gpt2_bpe_dropout0.3"):
        v = np.array([metrics("tokenizer_fix", n, s)["val_bpb"] for s in TOK_SEEDS])
        fix[n] = {"final_bpb": describe(v), "vs_gpt2_final": paired(gpt2, v), "vs_gpt2_best": paired(best_gpt2, v),
                  "vs_char_final": paired(final["char"], v),
                  "best_bpb": describe([min(e["bpb"] for e in eval_log("tokenizer_fix", n, s)) for s in TOK_SEEDS]),
                  "curve": curve("tokenizer_fix", n, TOK_SEEDS)}
    res["fix"] = fix
    # the fixes are compared with the 2000-step GPT-2 runs and with char: their own noise floor
    res["fix_noise_floor"] = noise_floor([gpt2, final["char"]] + [np.array(fix[n]["final_bpb"]["values"]) for n in fix])
    return res


def recipe():
    names = RECIPE + RECIPE_PARTS
    v = {n: np.array([metrics("recipe", n, s)["val_bpb"] for s in RECIPE_SEEDS]) for n in names}
    m0 = {n: metrics("recipe", n, 1337) for n in names}
    res = {"configs": {n: {**describe(v[n]), "params": m0[n]["params"], "params_non_embedding": m0[n]["params_non_embedding"],
                           "tokens_seen": m0[n]["tokens_seen"], "steps": m0[n]["steps"],
                           "eval_tokens": m0[n]["val_tokens"],
                           "train_min": float(np.mean([metrics("recipe", n, s)["train_time_sec"] for s in RECIPE_SEEDS])) / 60,
                           "curve": curve("recipe", n, RECIPE_SEEDS)} for n in names},
           "noise_floor": noise_floor(v.values())}
    chain = ["gpt2_textbook", "plus_rmsnorm_rope", "plus_wsd", "plus_muon"]
    res["chain"] = {f"{a}->{b}": paired(v[a], v[b]) for a, b in zip(chain, chain[1:])}
    res["chain"]["gpt2_textbook->plus_muon"] = paired(v["gpt2_textbook"], v["plus_muon"])
    # the position encoding of the textbook model: none at all, and learned against RoPE at 4x the tokens
    pv = {n: np.array([metrics("positions", n, s)["val_bpb"] for s in RECIPE_SEEDS])
          for n in ("textbook_nope", "textbook_steps4000", "textbook_rope_steps4000")}
    res["positions"] = {"configs": {n: {**describe(pv[n]), "params": metrics("positions", n, 1337)["params"],
                                        "tokens_seen": metrics("positions", n, 1337)["tokens_seen"],
                                        "curve": curve("positions", n, RECIPE_SEEDS)} for n in pv},
                        "nope_vs_learned": paired(v["gpt2_textbook"], pv["textbook_nope"]),
                        "nope_vs_rope": paired(v["textbook_rope"], pv["textbook_nope"]),
                        "rope_vs_learned_4000": paired(pv["textbook_steps4000"], pv["textbook_rope_steps4000"]),
                        "noise_floor": noise_floor(pv.values())}
    # the first step split: each architecture knob alone on the textbook model
    res["parts"] = {"rope_alone": paired(v["gpt2_textbook"], v["textbook_rope"]),
                    "rmsnorm_alone": paired(v["gpt2_textbook"], v["textbook_rmsnorm"]),
                    "rmsnorm_and_no_bias_after_rope": paired(v["textbook_rope"], v["plus_rmsnorm_rope"]),
                    # the Muon runs also raise the embedding learning rate and drop weight decay; adamw_matched has
                    # those settings with AdamW, so it splits the +Muon step into the two
                    "muon_settings_with_adamw": paired(v["plus_wsd"], v["adamw_matched"]),
                    "optimizer_matched": paired(v["adamw_matched"], v["plus_muon"])}
    # 2x2: schedule (cosine, WSD) x optimizer (AdamW, Muon) on the RMSNorm/RoPE model
    sched_adamw = v["plus_wsd"] - v["plus_rmsnorm_rope"]
    sched_muon = v["plus_muon"] - v["muon_cosine"]
    res["factorial"] = {"schedule_under_adamw": paired(v["plus_rmsnorm_rope"], v["plus_wsd"]),
                        "schedule_under_muon": paired(v["muon_cosine"], v["plus_muon"]),
                        "optimizer_under_cosine": paired(v["plus_rmsnorm_rope"], v["muon_cosine"]),
                        "optimizer_under_wsd": paired(v["plus_wsd"], v["plus_muon"]),
                        "interaction": paired(sched_adamw, sched_muon)}
    return res


def recipe_lr():
    """The AdamW configurations at lr 1e-3 (recipe sweep), 2e-3, 4e-3 and 8e-3 (recipe_lr sweep)."""
    out = {}
    for n in ("gpt2_textbook", "plus_rmsnorm_rope", "plus_wsd"):
        runs = {"1e-3": ("recipe", n), "2e-3": ("recipe_lr", f"{n}_lr2e-3"), "4e-3": ("recipe_lr", f"{n}_lr4e-3"),
                "8e-3": ("recipe_lr", f"{n}_lr8e-3")}
        v = {lr: np.array([metrics(sw, rn, s)["val_bpb"] for s in RECIPE_SEEDS]) for lr, (sw, rn) in runs.items()}
        best = min(v, key=lambda lr: v[lr].mean())
        out[n] = {"by_lr": {lr: describe(x) for lr, x in v.items()}, "best_lr": best, "best_values": v[best].tolist(),
                  "best_vs_other": {lr: paired(v[best], x) for lr, x in v.items() if lr != best}}
    muon = np.array([metrics("recipe", "plus_muon", s)["val_bpb"] for s in RECIPE_SEEDS])
    b = {n: np.array(out[n]["best_values"]) for n in out}
    res = {"configs": out,
           "noise_floor": noise_floor([np.array(x["values"]) for c in out.values() for x in c["by_lr"].values()]),
           "chain_best_lr": {"gpt2_textbook->plus_rmsnorm_rope": paired(b["gpt2_textbook"], b["plus_rmsnorm_rope"]),
                             "plus_rmsnorm_rope->plus_wsd": paired(b["plus_rmsnorm_rope"], b["plus_wsd"]),
                             "plus_wsd->plus_muon": paired(b["plus_wsd"], muon)}}
    return res


def scaling():
    pts = {}
    for w in WIDTHS:
        for st in STEPS:
            sweep, name = scaling_run(w, st)
            ms = [metrics(sweep, name, s) for s in PAIR_SEEDS]
            n_ne, d = ms[0]["params_non_embedding"], ms[0]["tokens_seen"]
            pts[f"{w}_{st}"] = {"width": w, "steps": st, "params": ms[0]["params"], "params_non_embedding": n_ne,
                                "tokens": d, "flops_6ND": 6 * n_ne * d, **describe([m["val_bpb"] for m in ms])}
    depth = {}
    for L in DEPTHS:
        for st in STEPS:
            sweep, name = scaling_run(256, st) if L == 4 else ("scaling", f"d256_L{L}_steps{st}")
            ms = [metrics(sweep, name, s) for s in PAIR_SEEDS]
            n_ne, d = ms[0]["params_non_embedding"], ms[0]["tokens_seen"]
            depth[f"{L}_{st}"] = {"width": 256, "n_layers": L, "steps": st, "params": ms[0]["params"],
                                  "params_non_embedding": n_ne, "tokens": d, "flops_6ND": 6 * n_ne * d,
                                  **describe([m["val_bpb"] for m in ms])}
    groups = [p["values"] for p in pts.values()] + [p["values"] for k, p in depth.items() if not k.startswith("4_")]
    res = {"points": pts, "depth_points": depth, "noise_floor": noise_floor(groups)}
    # the depth series (width 256) against the width series (4 layers) at about the same non-embedding parameters
    res["depth_vs_width"] = {st: [{"depth_series": f"256x{Ld}", "width_series": f"{w}x4",
                                   "param_ratio": depth[f"{Ld}_{st}"]["params_non_embedding"] /
                                   pts[f"{w}_{st}"]["params_non_embedding"],
                                   **paired(pts[f"{w}_{st}"]["values"], depth[f"{Ld}_{st}"]["values"])}
                                  for Ld, w in ((2, 176), (8, 352))] for st in STEPS}
    # gain per width step (about 2x N) at each budget, and 2x model against 2x tokens at about equal compute
    res["size_steps"] = {st: [paired(pts[f"{a}_{st}"]["values"], pts[f"{b}_{st}"]["values"]) |
                              {"from": a, "to": b, "param_ratio": pts[f"{b}_{st}"]["params_non_embedding"] /
                               pts[f"{a}_{st}"]["params_non_embedding"]}
                              for a, b in zip(WIDTHS, WIDTHS[1:])] for st in STEPS}
    # the same gains per doubling of N (the width steps multiply N by 1.89 to 2.12)
    for st in STEPS:
        for x in res["size_steps"][st]:
            x["gain_per_doubling"] = -x["mean_diff"] / np.log2(x["param_ratio"])
    res["token_steps"] = {w: [paired(pts[f"{w}_{a}"]["values"], pts[f"{w}_{b}"]["values"]) | {"from": a, "to": b}
                              for a, b in zip(STEPS, STEPS[1:])] for w in WIDTHS}
    # the same comparison along the depth series: 2 -> 4 and 4 -> 8 layers double N at width 256
    res["depth_model_vs_tokens"] = [{"small": a, "big": b, "steps": st,
                                     "flops_ratio": depth[f"{a}_{2 * st}"]["flops_6ND"] / depth[f"{b}_{st}"]["flops_6ND"],
                                     "bigger_model": depth[f"{b}_{st}"]["mean"], "more_tokens": depth[f"{a}_{2 * st}"]["mean"],
                                     "more_tokens_minus_bigger_model": paired(depth[f"{b}_{st}"]["values"],
                                                                              depth[f"{a}_{2 * st}"]["values"])}
                                    for st in STEPS[:-1] for a, b in ((2, 4), (4, 8))]
    res["model_vs_tokens"] = []
    for st in STEPS[:-1]:
        for a, b in zip(WIDTHS, WIDTHS[1:]):
            big, more = pts[f"{b}_{st}"], pts[f"{a}_{2 * st}"]
            res["model_vs_tokens"].append({"small": a, "big": b, "steps": st, "flops_bigger_model": big["flops_6ND"],
                                           "flops_more_tokens": more["flops_6ND"],
                                           "bigger_model": big["mean"], "more_tokens": more["mean"],
                                           "more_tokens_minus_bigger_model": paired(big["values"], more["values"])})
    return res


def swiglu():
    names = ["gelu", "swiglu", "gelu_lr5e-4", "swiglu_lr5e-4", "gelu_lr2e-3", "swiglu_lr2e-3", "gelu_lr4e-3",
             "swiglu_lr4e-3", "gelu_steps6000", "swiglu_steps6000"]
    v = {n: np.array([metrics("swiglu", n, s)["val_bpb"] for s in SWIGLU_SEEDS]) for n in names}
    m0 = {n: metrics("swiglu", n, 1337) for n in names}
    res = {"configs": {n: {**describe(v[n]), "params": m0[n]["params"], "tokens_seen": m0[n]["tokens_seen"],
                           "eval_tokens": m0[n]["val_tokens"], "curve": curve("swiglu", n, SWIGLU_SEEDS)} for n in names},
           "noise_floor": noise_floor(v.values()),
           "published": {"plus_rmsnorm_rope (gelu)": 1.744, "plus_swiglu": 1.786, "plus_swiglu_lr5e-4": 1.814}}
    res["swiglu_minus_gelu"] = {"lr1e-3": paired(v["gelu"], v["swiglu"]),
                                "lr5e-4": paired(v["gelu_lr5e-4"], v["swiglu_lr5e-4"]),
                                "lr2e-3": paired(v["gelu_lr2e-3"], v["swiglu_lr2e-3"]),
                                "lr4e-3": paired(v["gelu_lr4e-3"], v["swiglu_lr4e-3"]),
                                "steps6000": paired(v["gelu_steps6000"], v["swiglu_steps6000"])}
    # each MLP at its own best learning rate (lowest mean over the four rates)
    lrs = {"gelu": ["gelu_lr5e-4", "gelu", "gelu_lr2e-3", "gelu_lr4e-3"],
           "swiglu": ["swiglu_lr5e-4", "swiglu", "swiglu_lr2e-3", "swiglu_lr4e-3"]}
    best = {k: min(ns, key=lambda n: v[n].mean()) for k, ns in lrs.items()}
    res["best_lr"] = best
    res["swiglu_minus_gelu"]["best_vs_best"] = paired(v[best["gelu"]], v[best["swiglu"]])
    # where the mean curves cross: last evaluation at which SwiGLU is still ahead of GELU
    for tag, (g, s) in {"lr1e-3": ("gelu", "swiglu"), "steps6000": ("gelu_steps6000", "swiglu_steps6000")}.items():
        cg, cs = res["configs"][g]["curve"], res["configs"][s]["curve"]
        ahead = [t for t, a, b in zip(cg["tokens"], cg["mean"], cs["mean"]) if b < a]
        res[f"swiglu_ahead_until_{tag}"] = max(ahead) if ahead else None
    return res


def main():
    out = {"tokenizers": tokenizers(), "recipe": recipe(), "recipe_lr": recipe_lr(), "scaling": scaling(),
           "swiglu": swiglu()}
    meta = json.loads((ROOT / "runs" / "sweeps" / "recipe" / "plus_muon_s1337" / "meta.json").read_text())
    out["nanolm_git"], out["torch"] = meta["git"], meta["torch"]
    allm = list((ROOT / "runs" / "sweeps").rglob("metrics.json"))
    out["n_runs"] = len(allm)
    out["total_train_hours"] = math.fsum(json.loads(p.read_text())["train_time_sec"] for p in allm) / 3600
    out["devices"] = sorted({json.loads(p.read_text())["device"] for p in allm})
    save("summary", out)
    print(json.dumps({k: out[k] for k in ("n_runs", "total_train_hours", "devices")}, indent=1))


if __name__ == "__main__":
    main()
