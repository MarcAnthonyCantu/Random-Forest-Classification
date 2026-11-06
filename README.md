# Image Classification on CIFAR-10 with Random Forests

A study of Random Forest classifiers on a three-class subset of CIFAR-10
(cat, frog, truck), going beyond a single accuracy number to probe *how* the
ensemble behaves: hyperparameter sensitivity, out-of-bag validation, color-channel
importance, the spread of individual tree accuracy, and whether a pruned forest
can match a full one.

## Overview

This project uses Random Forests as a lens for
understanding ensemble behavior on raw image pixels. A Random Forest reaches
**~75% test accuracy** on the three classes while individual trees average only
**~57%**, the analysis digs
into where that accuracy comes from and what it costs.

- **Data:** three CIFAR-10 classes (cat, frog, truck); each image is a 32×32
  RGB array flattened to 3,072 pixel features.
- **Model:** scikit-learn `RandomForestClassifier`.
- **Angle:** interpretation and diagnostics — overfitting, parameter and
  ensemble-size sensitivity, channel ablation, per-tree accuracy, and pruning.

## Method & findings

1. **Baseline & overfitting** — A 100-tree forest reaches **74.3%** test
   accuracy at **100%** training accuracy.
2. **Hyperparameter sweep** — Split criterion, tree depth, min-samples-split /
   -leaf, max-samples, and max-features are each varied. Differences are small
   (e.g. `gini` 74.27% vs. `entropy` 74.22%), indicating the model is not very
   sensitive to these on this data.
3. **Out-of-bag validation** — An OOB forest gives an OOB estimate of
   **72.6%** against a test accuracy of **75.2%**, showing OOB as a usable
   internal validation signal without a separate holdout.
4. **Ensemble size** — Forests of 100/200/400/600 trees land at
   75.2% / 74.4% / 75.2% / 74.7% — accuracy plateaus past ~100 trees, so added
   trees buy computation without accuracy.
5. **Color-channel ablation** — Dropping each RGB channel costs roughly 4–5%
   accuracy (red most, then green, then blue), quantifying how much color
   carries the signal for these classes.
6. **Per-tree accuracy** — Individual trees span **54–59%** (mean ~57%), yet the
   100-tree ensemble reaches ~75%.
7. **Pruning** — Removing the lowest-accuracy 20% of trees changes test accuracy
   by only **−0.18%** (75.16% → 74.98%), so a 20%-smaller forest is
   near-equivalent.

## Repository contents

| File | Description |
|------|-------------|
| `code.ipynb` | Full analysis — loading, class selection, training, tuning, and the diagnostics above |
| `Report.pdf` | Written report with figures (confusion matrices, tree-accuracy plots) and discussion |

## Tech stack

- **Python** — NumPy, pandas, matplotlib
- **scikit-learn** — `RandomForestClassifier`, model-selection and metrics utilities
- Jupyter Notebook

## Running it

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
jupyter notebook code.ipynb
```

The notebook loads CIFAR-10 from a local `cifar-10-batches-py/` directory (the
standard Python version of the dataset from
<https://www.cs.toronto.edu/~kriz/cifar.html>). Download and extract it into the
repository root before running; it is not redistributed here.

## Background

This project was developed for a graduate course in mathematical machine
learning. The emphasis is on interpreting model behavior — overfitting,
ensemble effects, feature importance, and the accuracy/efficiency trade-off
rather than on maximizing a single accuracy score.
