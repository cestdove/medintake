# MedIntake — Model Benchmark

## Dataset

**Medical Abstracts Text Classification Dataset**

| Property | Value |
|---|---:|
| Training samples | 11,550 |
| Test samples | 2,888 |
| Number of classes | 5 |
| Original labels | 1–5 |

---

## Results

| Model | Features / Architecture | Epochs | Batch Size | Learning Rate | Accuracy | Macro F1 | Weighted F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| LinearSVC | TF-IDF | — | — | — | 50.45% | ~0.50 | — |
| BERT | `bert-base-uncased` | 1 | 8 | `2e-5` | 64.37% | 0.64 | 0.62 |
| BERT | `bert-base-uncased` | 3 | 8 | `2e-5` | **65.17%** | **0.65** | **0.63** |
| BERT + LR Scheduler | `bert-base-uncased` | 3 | 8 | `2e-5` | 63.12% | 0.63 | 0.62 |
| RoBERTa | `roberta-base` | 3* | 8 | `2e-5` | — | — | — |

\* Training was interrupted during the second epoch; no final evaluation metrics are reported.

---

## Experimental Setup

Transformer experiments use the following general configuration unless otherwise specified:

- Optimizer: **AdamW**
- Batch size: **8**
- Learning rate: **`2e-5`**
- Device: **MPS**
- Number of classes: **5**

The experiments are designed to compare Transformer architectures while keeping the main training configuration consistent.

---

# Experiment 1 — BERT, 1 Epoch

## Configuration

- Model: `bert-base-uncased`
- Epochs: 1
- Batch size: 8
- Optimizer: AdamW
- Learning rate: `2e-5`
- Device: MPS
- Training batches: 1,444
- Average training loss: `1.0173`

## Results

- Accuracy: **64.37%**
- Macro F1: **0.64**
- Weighted F1: **0.62**

## Classification Report

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| 0 | 0.69 | 0.86 | 0.77 | 633 |
| 1 | 0.54 | 0.75 | 0.63 | 299 |
| 2 | 0.60 | 0.65 | 0.63 | 385 |
| 3 | 0.67 | 0.81 | 0.74 | 610 |
| 4 | 0.65 | 0.36 | 0.46 | 961 |

## Confusion Matrix

| Actual / Predicted | 0 | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|---:|
| **0** | 546 | 17 | 18 | 7 | 45 |
| **1** | 38 | 225 | 8 | 7 | 21 |
| **2** | 33 | 10 | 251 | 27 | 64 |
| **3** | 18 | 15 | 24 | 495 | 58 |
| **4** | 156 | 148 | 115 | 200 | 342 |

## Observation

BERT achieved **64.37% accuracy after one epoch**, compared with approximately **50.45%** for the TF-IDF + LinearSVC baseline.

Class 4 remained the main source of errors, with a recall of **0.36** and an F1-score of **0.46**.

---

# Experiment 2 — BERT, 3 Epochs

## Configuration

- Model: `bert-base-uncased`
- Epochs: 3
- Batch size: 8
- Optimizer: AdamW
- Learning rate: `2e-5`
- Device: MPS
- Training batches: 4,332
- Average training loss: `0.8401`

## Results

- Accuracy: **65.17%**
- Macro F1: **0.65**
- Weighted F1: **0.63**

## Classification Report

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| 0 | 0.67 | 0.89 | 0.77 | 633 |
| 1 | 0.57 | 0.65 | 0.61 | 299 |
| 2 | 0.61 | 0.64 | 0.62 | 385 |
| 3 | 0.67 | 0.85 | 0.75 | 610 |
| 4 | 0.68 | 0.38 | 0.49 | 961 |

## Confusion Matrix

| Actual / Predicted | 0 | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|---:|
| **0** | 562 | 13 | 15 | 9 | 34 |
| **1** | 49 | 194 | 5 | 10 | 41 |
| **2** | 43 | 12 | 245 | 29 | 56 |
| **3** | 18 | 8 | 25 | 517 | 42 |
| **4** | 164 | 113 | 113 | 207 | 364 |

## Observation

Increasing training from one to three epochs reduced the average training loss from `1.0173` to `0.8401`.

However, the improvement on the test set was relatively small:

| Metric | 1 Epoch | 3 Epochs | Change |
|---|---:|---:|---:|
| Accuracy | 64.37% | **65.17%** | +0.80 pp |
| Macro F1 | 0.64 | **0.65** | +0.01 |
| Weighted F1 | 0.62 | **0.63** | +0.01 |

Class 4 remained the main source of errors. Its recall increased from `0.36` to `0.38`, while its F1-score increased from `0.46` to `0.49`.

The results suggest that simply increasing the number of fine-tuning epochs provides diminishing returns with the current BERT training configuration.

---

# Experiment 3 — BERT + Dynamic Padding + LR Scheduler

## Configuration

- Model: `bert-base-uncased`
- Epochs: 3
- Batch size: 8
- Optimizer: AdamW
- Initial learning rate: `2e-5`
- Learning-rate scheduler: linear decay
- Warmup: 10% of training steps
- Dynamic padding: enabled
- Device: MPS
- Training batches: 4,332
- Average training loss: `0.8493`

## Results

- Accuracy: **63.12%**
- Macro F1: **0.63**
- Weighted F1: **0.62**

## Classification Report

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| 0 | 0.68 | 0.83 | 0.75 | 633 |
| 1 | 0.53 | 0.60 | 0.56 | 299 |
| 2 | 0.60 | 0.60 | 0.60 | 385 |
| 3 | 0.67 | 0.81 | 0.73 | 610 |
| 4 | 0.60 | 0.41 | 0.49 | 961 |

## Confusion Matrix

| Actual / Predicted | 0 | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|---:|
| **0** | 523 | 21 | 20 | 16 | 53 |
| **1** | 42 | 179 | 8 | 8 | 62 |
| **2** | 35 | 11 | 231 | 32 | 76 |
| **3** | 15 | 10 | 21 | 495 | 69 |
| **4** | 154 | 114 | 106 | 192 | 395 |

## Observation

Adding dynamic padding, a linear learning-rate scheduler, and a 10% warmup period did not improve overall test performance.

Compared with the three-epoch BERT configuration using a constant learning rate:

| Metric | Constant LR | Scheduler + Warmup | Change |
|---|---:|---:|---:|
| Accuracy | **65.17%** | 63.12% | -2.05 pp |
| Macro F1 | **0.65** | 0.63 | -0.02 |
| Weighted F1 | **0.63** | 0.62 | -0.01 |

Class 4 recall increased from `0.38` to `0.41`, but its F1-score remained at `0.49`. Performance decreased across several of the other classes, particularly classes 1 and 2.

The experiment suggests that the linear learning-rate schedule with 10% warmup did not provide an advantage for the current fine-tuning setup.

The three-epoch BERT configuration with a constant `2e-5` learning rate remains the best-performing configuration tested so far.

---

# Experiment 4 — RoBERTa

## Configuration

- Model: `roberta-base`
- Planned epochs: 3
- Batch size: 8
- Optimizer: AdamW
- Learning rate: `2e-5`
- Device: MPS
- Dynamic padding: enabled
- Learning-rate scheduler: none
- Training batches per epoch: 1,444

## Status

The RoBERTa experiment was started using the same main training configuration as the best-performing BERT experiment.

The tokenizer and `RobertaForSequenceClassification` model initialized successfully with five output classes, and training started successfully on MPS.

The run was interrupted during the **second epoch** because training throughput was substantially slower than the previous BERT experiments on the current hardware.

### Training Progress Before Interruption

**Epoch 1/3**

- Batch 1,400/1,444 reached
- Final logged batch loss: `0.7349`

**Epoch 2/3**

- Batch 800/1,444 reached
- Final logged batch loss before interruption: `0.5185`

The logged batch losses are not used as final evaluation metrics because they represent individual training batches rather than performance on the held-out test set.

### Evaluation

No final test evaluation was performed for this run.

Therefore, **no accuracy, Macro F1, Weighted F1, or confusion matrix is reported for RoBERTa**.

---

## Current Findings

The experiments so far show a substantial improvement when moving from the TF-IDF baseline to Transformer-based fine-tuning.

The best completed configuration is currently:

**BERT (`bert-base-uncased`) — 3 epochs — `2e-5` constant learning rate**

with:

- Accuracy: **65.17%**
- Macro F1: **0.65**
- Weighted F1: **0.63**

Increasing BERT training from one to three epochs produced only a modest improvement, while the tested scheduler + warmup configuration performed worse on the held-out test set.

RoBERTa was successfully initialized and trained, but its run was interrupted before evaluation because of substantially slower training throughput on the current hardware.

---

## Next Experiments

Planned directions include:

- Benchmarking more recent Transformer architectures.
- Evaluating biomedical-domain pretrained language models.
- Performing deeper Transformer error analysis.
- Comparing general-domain and biomedical-domain representations.
- Exploring retrieval-based approaches and Retrieval-Augmented Generation (RAG).
- Evaluating retrieval quality separately from generation quality.