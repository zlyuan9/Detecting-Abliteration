# Detecting Abliteration

Code and saved results for *Detecting Abliteration as a Projection of the Base Weights*.

Abliteration replaces each residual-writing matrix `W` by `W - alpha R R^T W`. The weight
difference is therefore a scaled projection of the base weights. The test measures how far an
observed difference is from such a projection (the scanned projection residual).

## Layout

| File | Purpose |
|---|---|
| `experiments/mech.py` | The statistic (`resid_scan`) |
| `experiments/d1_deltastruct.py` | Shared helpers: shard handling, residual-writing tensor selection |
| `experiments/d6_sweep.py` | Scores public derivatives of a base model (downloads weights) |
| `experiments/label_from_cards.py` | Writes `results/labels.json` from model cards |
| `experiments/synth_scan.py` | Controlled edits of Qwen3-0.6B (downloads ~2.7 GB, ~9 min on CPU) |
| `experiments/partial_base.py`, `experiments/cross_author.py` | Leading difference directions and cross-author agreement |
| `experiments/labels.py` | Loads per-model scores and labels |
| `experiments/plot_paper.py` | Figures, per-model table, and the numbers behind the paper's tables |
| `experiments/results/` | Saved per-model scores (`d6_*.json`), labels, and agreement results |

## Reproduce the paper's figures and tables

No downloads and no GPU are needed:

```
pip install -r requirements.txt
python experiments/plot_paper.py
```

This writes `paper/figures/fig_scan.pdf`, `fig_resid.pdf`, `fig_resid_main.pdf` and
`paper/tab_models.tex`, and prints the numbers used in the hardening, fixed-k, agreement and baseline-comparison
tables.

## Re-score the models

`d6_sweep.py` takes `--base`, `--tag` and `--derivs` (a file with one Hugging Face repo id per
line) and writes `results/d6_<tag>.json`. It downloads weights and runs on CPU. The repo ids of
every model evaluated are in `results/d6_*.json` and listed in the paper's appendix.

The agreement results in `results/xauthor_Qwen3-0_6B.json` come from `partial_base.py`
followed by `cross_author.py`. The intermediate direction files are large and are not included.
