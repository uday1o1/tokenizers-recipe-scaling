# Tokenizers, the training recipe and scaling for small GPT language models

Controlled experiments on small GPT models that separate the effect of an idea from the effect of its training budget:

1. **Tokenizers.** Eight tokenizers on Tiny Shakespeare at a fixed number of steps: why per-token loss cannot compare them, how tokenizers with long tokens overfit, and a fix for it.
2. **The training recipe.** On FineWeb, from a GPT-2-style model to RMSNorm + RoPE, the WSD schedule and the Muon optimizer, one change at a time. A 2 × 2 of schedule × optimizer separates the two, and a further sweep (`positions`) isolates the position encoding.
3. **Scaling.** Five widths and four depths at three token budgets, plotted against non-embedding parameters and against training FLOPs: does the curve flatten, and is a 2× larger model or 2× more tokens the better use of compute?
4. **SwiGLU against GELU.** At nanolm's 6 × 384 size, where SwiGLU loses: is it the learning rate or the token budget?

All training uses [nanolm](https://github.com/lkk688/DeepDataMiningLearning/tree/main/DeepDataMiningLearning/llm) at commit `8418fc1`, unmodified. Every experiment is a sweep file in `configs/sweeps/`, and every statistic below is computed from the run records in `runs/` by `analysis/summarize.py` into `results/summary.json` or is a span, difference, ratio or count of its values (the size of the validation split comes from the byte and char runs' `stdout.log`, and the attention term of the compute ratios, 12 L d T FLOPs per token with T = 512, from each run's `config.yaml`); values attributed to nanolm's tutorial are quoted from it.

## Results

Results are validation bits per byte (bpb; lower is better), as mean ± std over seeds, unless a column says otherwise. Differences are paired over seeds, and a difference below twice the seed noise floor of its study (median within-configuration std) is treated as not resolved.

### Tokenizers (3 seeds, 2000 steps × 8,192 tokens)

| tokenizer | vocabulary | bytes per token | passes over the training text | loss (nats/token) | final bpb | best bpb (step per seed) |
|---|---:|---:|---:|---:|---:|---:|
| word 2048 | 2,048 | 1.62 | 29.2 | 2.33 | 2.070 ± 0.011 | 2.039 ± 0.008 (1250, 1500, 1250) |
| byte | 256 | 1.00 | 16.3 | 1.45 | 2.092 ± 0.003 | 2.089 ± 0.008 (1750, 2000, 2000) |
| char | 65 | 1.00 | 16.3 | 1.47 | 2.115 ± 0.010 | 2.111 ± 0.006 (1750, 2000, 2000) |
| BPE 512 | 512 | 1.76 | 29.3 | 2.62 | 2.152 ± 0.005 | 2.116 ± 0.010 (1250, 1250, 1000) |
| word 8192 | 8,192 | 1.98 | 36.6 | 3.16 | 2.308 ± 0.003 | 2.109 ± 0.015 (750, 750, 500) |
| unigram 2048 | 2,048 | 2.36 | 42.0 | 3.80 | 2.327 ± 0.007 | 2.108 ± 0.002 (500, 500, 500) |
| BPE 2048 | 2,048 | 2.41 | 43.4 | 3.96 | 2.371 ± 0.004 | 2.113 ± 0.004 (500, 500, 500) |
| GPT-2 BPE | 50,257 | 3.09 | 54.3 | 6.14 | 2.862 ± 0.075 | 2.166 ± 0.006 (250, 250, 500) |

- **Per-token loss ranks the tokenizers by token length, not by quality.** Loss orders the eight exactly by bytes per token. Word 2048 has a far higher loss than char (2.33 against 1.47 nats per token) but a lower bpb, because each of its tokens covers 1.62 bytes. bpb = loss / (ln 2 × bytes per token) charges every tokenizer per byte of the same validation split (each tokenizer's evaluation windows cover 110,827 to 111,403 of the 111,540 bytes of the validation split: `val_bytes` in each run's `metrics.json`, against the validation size the byte and char runs print in `stdout.log`).
- **Long-token tokenizers overfit.** Word 8192, unigram 2048, BPE 2048 and GPT-2 BPE (1.98 to 3.09 bytes per token) pass over the 1 MB corpus 36.6 to 54.3 times in the same number of steps. They reach their best bpb by step 750 and then get worse, and the gap between final and best rises with the passes. The best points of all eight tokenizers span 0.127 bpb, the final values 0.792.
- **Fixing GPT-2 BPE.** A 500-step schedule (the step at which nanolm's tutorial reports this tokenizer's best bpb) gives 2.146 ± 0.018, −0.716 from the 2000-step final (paired p = 0.005). Dropout 0.3 gives 2.344 ± 0.021, slowing the overfitting without stopping it.

### The training recipe (FineWeb, 4 layers × 256, 16.4M tokens, 5 seeds)

| configuration | parameters | final bpb | paired change |
|---|---:|---:|---:|
| GPT-2 textbook (LayerNorm, learned positions, biases; AdamW, cosine) | 3,683,840 | 2.491 ± 0.003 | |
| + RMSNorm, RoPE, no biases | 3,410,176 | 2.124 ± 0.003 | −0.368 |
| + WSD schedule (AdamW) | 3,410,176 | 2.066 ± 0.002 | −0.058 |
| + Muon (WSD) | 3,410,176 | 1.831 ± 0.005 | −0.235 |
| control: Muon, cosine | 3,410,176 | 1.845 ± 0.007 | −0.278 against + RMSNorm, RoPE |
| control: AdamW, WSD, with the Muon runs' embedding learning rate and no weight decay | 3,410,176 | 2.058 ± 0.006 | −0.008 against + WSD (not resolved) |

- Each step of the chain helps on all 5 seeds (paired p < 0.0001). At this budget the architecture step is the largest; in nanolm's tutorial at 6 × 384 and 49M tokens it is the smallest (−0.046, against −0.058 for WSD and −0.081 for Muon, in a chain that adds SwiGLU before WSD).
- **Muon's gain is almost all the optimizer.** The Muon runs also raise the embedding learning rate, drop weight decay and train the norm gains at 3e-3 instead of 1e-3. The first two settings alone, with AdamW, give −0.008 (not resolved), and Muon against that matched AdamW run gives −0.227 (all 5 seeds); only the norm gains' learning rate stays unmatched.
- **Learning rate.** The three AdamW configurations of the chain each have their lowest mean bpb at lr 4e-3 (of 1e-3, 2e-3, 4e-3 and 8e-3; `recipe_lr`), ahead of 8e-3 by 0.021, 0.007 and 0.016 (the 0.007 not resolved: for the RMSNorm/RoPE configuration 8e-3 is tied with 4e-3; with it at 8e-3 the WSD step is −0.026). Each at its lowest-mean rate, the steps become −0.340 (architecture), −0.019 (WSD) and −0.181 (Muon, at its own settings), all resolved and in the same order.
- **Schedule against optimizer.** Muon helps under either schedule (−0.278 under cosine, −0.235 under WSD), and WSD under either optimizer (−0.058 with AdamW, −0.015 with Muon). The two interact (+0.043, p < 0.0001), so the gain credited to Muon depends on the schedule it is compared under.
- **It is the positions.** RoPE alone on the textbook model gives −0.377. RMSNorm alone gives +0.001 (not resolved), and replacing LayerNorm with RMSNorm and dropping the biases on the RoPE model costs +0.010. With no position encoding at all the textbook model reaches 2.507, only 0.016 worse than learned positions, so the learned table adds little after 16.4M tokens. With 4× the tokens (65.5M) the gap between learned positions and RoPE shrinks from 0.377 to 0.025.

### Scaling (2 seeds)

- **Trend.** Along the width series bpb falls with every width step at every budget, and by these point estimates the gain per doubling of non-embedding parameters N shrinks at every step: from 0.092 to 0.031 bpb at 16.4M tokens and from 0.073 to 0.038 at 65.5M (each width step multiplies N by 1.89 to 2.12). The curve flattens, less so with more tokens.
- **Shape.** 8 layers × 256 (5.8% more non-embedding parameters than 4 layers × 352) is lower than 4 × 352 by 0.005 to 0.015 bpb across the budgets (all 3 resolved). 2 layers × 256 (5.8% more non-embedding parameters than 4 layers × 176) is 0.020 to 0.069 worse than 4 × 176.
- **2× model or 2× tokens.** At about equal compute (6ND within 6%; with attention counted the twice-the-tokens point has 1.01 to 1.19 times its pair's compute, and in the 4 pairs within 6% it still gains 0.008 to 0.058), a width-series model with twice the tokens beats the next width up in 7 of 8 pairs, by 0.008 to 0.058 bpb. The eighth pair (−0.002) is not resolved; its larger model has the most tokens per non-embedding parameter of all pairs (22). A doubling of tokens gains 0.068 to 0.114 bpb at every width, a doubling of N 0.031 to 0.092. Along the depth series the same holds from 4 to 8 layers, but 2 layers on twice the tokens is not better than 4 layers.

### SwiGLU against GELU (6 × 384, 49M tokens, 5 seeds)

- At nanolm's settings SwiGLU is worse, 1.781 ± 0.021 against 1.753 ± 0.024 (+0.028, paired p = 0.08, 4 of 5 seeds; the tutorial reports 1.786 and 1.744).
- **Learning rate.** Over learning rates 5e-4 to 4e-3, SwiGLU leads only at 5e-4, where GELU is far from its best. At each MLP's lowest-mean rate of the four (GELU 4e-3, the top of the range; SwiGLU 2e-3) GELU leads by 0.061 (paired p = 0.009, all 5 seeds). SwiGLU's optimum is bracketed, so a wider grid could only lower GELU's best and widen its lead.
- **Token budget.** With twice the tokens it still trails, by 0.023 (all 5 seeds). At either budget SwiGLU's mean curve is ahead early on and falls behind after 24.6M tokens.
- Neither hypothesis explains the result at this size.

## Layout

| path | contents |
|---|---|
| `configs/sweeps/` | the seven sweep files: `tokenizers`, `tokenizer_fix`, `recipe`, `recipe_lr`, `positions`, `scaling`, `swiglu` |
| `runs/sweeps/<sweep>/<run>_s<seed>/` | nanolm's record of every run: `config.yaml`, `meta.json` (git commit, device, command), `train_log.jsonl`, `eval_log.jsonl`, `metrics.json`, `sample.txt`, `stdout.log` (no checkpoints) |
| `runs/sweeps/<sweep>/summary.{csv,md}`, `curves.png`, `bars.png`, `analyze_stdout.txt`, `sweep.yaml` | `python -m nanolm.analyze` output per sweep (with paired p against a baseline for `tokenizers`, `recipe`, `positions` and `swiglu`) and its printed table, and a copy of the sweep file (identical to the one in `configs/sweeps/`) |
| `analysis/` | `summarize.py` (every summary number and paired test into `results/summary.json`), `figures.py` (all figures), `common.py` (shared helpers) |
| `results/` | `summary.json` and `environment.txt` (machine and package versions) |
| `figures/` | `tokenizers.png`, `recipe.png`, `scaling.png`, `swiglu.png` |
| `scripts/` | `setup.sh` (clone nanolm, install requirements), `run_sweeps.sh` (train every run), `env.sh` (set PYTHONPATH) |
| `requirements.txt` | pinned package versions |

## Reproducing

Requires Python 3.11 to 3.13 and one NVIDIA GPU. Training used one RTX 4090 with bf16 autocast, PyTorch 2.6.0, CUDA 12.4 and Python 3.11; `requirements.txt` pins the package versions.

```bash
bash scripts/setup.sh          # clone nanolm at commit 8418fc1 into external/, install requirements
bash scripts/run_sweeps.sh     # download FineWeb (sp1024), train every run (finished runs are skipped), nanolm.analyze per sweep
source scripts/env.sh
python analysis/summarize.py   # results/summary.json (no GPU)
python analysis/figures.py     # figures/*.png (no GPU)
```

- **Time.** The sum of per-run training times is 13.3 hours. `SLOTS=6 bash scripts/run_sweeps.sh` runs six at a time on one 24 GB GPU (two for `tokenizer_fix`); the per-run times in `metrics.json` include that sharing.
- **Without a GPU.** `summarize.py` rebuilds `results/summary.json`, from which every statistic above is read or derived (the attention FLOPs with each run's `config.yaml`), and `figures.py` rebuilds every figure in `figures/`, from the committed `runs/`.

## Limitations

- **Data.** One corpus and one validation split per study. The length of the GPT-2 tokenizer's 500-step schedule comes from the step at which nanolm's tutorial reports its best bpb, on the same validation split.
- **Tuning.** nanolm's default learning rates, not tuned per width or for Muon; learning-rate grids only for the three AdamW configurations of the recipe chain and in the SwiGLU study.
- **Statistics.** 2 seeds for the scaling points, and paired t-tests without correction for multiple comparisons.
- **Scope.** The largest model has 12.6M non-embedding parameters and the longest runs see 98.3M training tokens.
