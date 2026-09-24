# MedIntake — Model Benchmark

## Dataset

**Medical Abstracts Text Classification Dataset**

| Property | Value |
|---|---:|
| Training samples | 11,550 |
| Test samples | 2,888 |
| Classes | 5 |
| Labels | 1–5 |

---

## Benchmark Summary

| Experiment | Model | LR | Epochs | Accuracy | Macro F1 | Weighted F1 |
|---|---|---:|---:|---:|---:|---:|
| Baseline | LinearSVC + TF-IDF | — | — | 50.45% | ~0.50 | — |
| 1 | BERT | 2e-5 | 1 | 64.37% | 0.64 | 0.62 |
| 2 | BERT | 2e-5 | 3 | **65.17%** | **0.65** | 0.63 |
| 3 | BERT + scheduler + warmup | 2e-5 | 3 | 63.12% | 0.63 | 0.62 |
| 4 | RoBERTa | 2e-5 | 3 | 63.61% | 0.63 | 0.61 |
| 5 | RoBERTa | 2e-5 | 3 | 61.98% | 0.62 | 0.61 |
| 6 | RoBERTa | 1e-5 | 3 | 62.98% | 0.63 | 0.61 |
| 7 | RoBERTa | 1e-5 | 2 | 63.99% | 0.64 | 0.63 |
| 8 | BERT | 1e-5 | 2 | 64.34% | 0.64 | 0.63 |
| 9 | BERT | 1e-5 | 3 | 61.39% | 0.61 | 0.61 |
| 10 | BERT | 2e-5 | 2 | 64.58% | 0.64 | 0.62 |
| 11 | BERT | 2e-5 | 3 | 62.47% | 0.62 | 0.61 |
| 12 | **BiomedBERT** | 2e-5 | 2 | **65.37%** | **0.65** | **0.64** |

Experiments 5–12 use the controlled validation-based protocol described below.

---

## Experimental Setup

Transformer experiments use:

- Optimizer: **AdamW**
- Batch size: **8**
- Maximum sequence length: **256**
- Device: **MPS**
- Dynamic padding
- 5 output classes

For the controlled experiments, 10% of the original training set is reserved as a **stratified validation split** with `random_state=42`. The held-out test set is not used during training or validation.

Models are evaluated using accuracy, Macro F1, Weighted F1, classification reports, and confusion matrices.

---

# Initial Benchmark

## Experiment 1 — BERT, 1 Epoch

`bert-base-uncased`, `2e-5`, 1 epoch.

- Accuracy: **64.37%**
- Macro F1: **0.64**
- Weighted F1: **0.62**
- Training loss: `1.0173`

BERT substantially improved over the TF-IDF + LinearSVC baseline of 50.45%. Class 4 was already the main source of errors, with recall of `0.36`.

---

## Experiment 2 — BERT, 3 Epochs

`bert-base-uncased`, `2e-5`, 3 epochs.

- Accuracy: **65.17%**
- Macro F1: **0.65**
- Weighted F1: **0.63**
- Training loss: `0.8401`

Increasing training from one to three epochs improved accuracy by only **0.80 percentage points**, indicating diminishing returns.

---

## Experiment 3 — BERT + Scheduler

`bert-base-uncased`, `2e-5`, 3 epochs, linear learning-rate decay with 10% warmup and dynamic padding.

- Accuracy: **63.12%**
- Macro F1: **0.63**
- Weighted F1: **0.62**

Performance decreased by **2.05 pp** compared with the constant-learning-rate BERT configuration.

---

## Experiment 4 — RoBERTa

`roberta-base`, `2e-5`, 3 epochs.

- Accuracy: **63.61%**
- Macro F1: **0.63**
- Weighted F1: **0.61**

RoBERTa performed below the three-epoch BERT configuration and required substantially longer training time on the current MPS hardware.

---

# Controlled Experiments

## Experiments 5–7 — RoBERTa

The validation-based experiments tested learning rate and training duration.

| Experiment | LR | Epochs | Accuracy | Macro F1 |
|---|---:|---:|---:|---:|
| 5 | 2e-5 | 3 | 61.98% | 0.62 |
| 6 | 1e-5 | 3 | 62.98% | 0.63 |
| 7 | 1e-5 | 2 | **63.99%** | **0.64** |

Reducing the learning rate and training for two epochs produced the best RoBERTa result in the controlled phase.

---

## Experiments 8–11 — BERT

| Experiment | LR | Epochs | Accuracy | Macro F1 |
|---|---:|---:|---:|---:|
| 8 | 1e-5 | 2 | 64.34% | 0.64 |
| 9 | 1e-5 | 3 | 61.39% | 0.61 |
| 10 | 2e-5 | 2 | **64.58%** | **0.64** |
| 11 | 2e-5 | 3 | 62.47% | 0.62 |

Experiment 10 produced the strongest general-domain BERT result under the validation-based protocol.

Increasing training from two to three epochs reduced performance at both tested learning rates. In Experiment 11, validation loss decreased through epoch 2 and increased during epoch 3 while training loss continued to decrease, providing evidence of overfitting.

---

# Experiment 12 — BiomedBERT

**Model:** `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext`

Configuration:

- Learning rate: `2e-5`
- Epochs: 2
- Batch size: 8
- Validation split: 10%
- Maximum sequence length: 256
- Dynamic padding
- AdamW
- MPS

### Results

- Accuracy: **65.37%**
- Macro F1: **0.65**
- Weighted F1: **0.64**
- Average training loss: `0.8137`
- Validation loss: `0.8233 → 0.7671`

Compared with the matched BERT configuration from Experiment 10:

| Metric | BERT | BiomedBERT | Difference |
|---|---:|---:|---:|
| Accuracy | 64.58% | **65.37%** | **+0.79 pp** |
| Macro F1 | 0.64 | **0.65** | +0.01 |
| Weighted F1 | 0.62 | **0.64** | +0.02 |

BiomedBERT achieved higher test performance under the same validation-based training configuration. Validation loss also continued to decrease during the second epoch.

This provides initial evidence that biomedical-domain pretraining can benefit this medical abstract classification task.

Class 4 remains the most difficult class, with recall of `0.38` and F1 of `0.49`.

---

## Current Findings

The experiments show a substantial improvement from the TF-IDF baseline to Transformer-based models:

```text
TF-IDF + LinearSVC       50.45%
          ↓
BERT                     65.17%
          ↓
BiomedBERT               65.37%
```

The strongest result under the controlled validation protocol is currently **BiomedBERT at 65.37% accuracy**, compared with **64.58%** for the matched general-domain BERT configuration.

The initial benchmark and controlled experiments use different training protocols, so their results should be interpreted within their respective experimental settings.

Across the Transformer experiments, **class 4 consistently has the lowest recall and remains the main classification challenge**.

---

## Next Experiments

Planned directions include:

- Comparing additional biomedical-domain pretrained models.
- Performing deeper Transformer error analysis.
- Comparing general-domain and biomedical-domain representations.
- Evaluating training duration for biomedical-domain models.
- Benchmarking more recent Transformer architectures.
- Exploring retrieval-based approaches and RAG.