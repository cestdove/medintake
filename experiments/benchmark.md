# MedIntake — Model Benchmark

## Dataset

Medical Abstracts Text Classification Dataset.

- Training samples: 11,550
- Test samples: 2,888
- Number of classes: 5
- Original labels: 1–5

---

## Results

| Model | Features | Epochs | Batch Size | Learning Rate | Accuracy | Macro F1 | Weighted F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| LinearSVC | TF-IDF | — | — | — | 50.45% | ~0.50 | — |
| BERT | `bert-base-uncased` | 1 | 8 | 2e-5 | 64.37% | 0.64 | 0.62 |
| BERT | `bert-base-uncased` | 3 | 8 | 2e-5 | **65.17%** | **0.65** | **0.63** |
| BERT + LR Scheduler | `bert-base-uncased` | 3 | 8 | 2e-5 | 63.12% | 0.63 | 0.62 |

---

## Experiment 1 — BERT, 1 Epoch

### Configuration

- Model: `bert-base-uncased`
- Epochs: 1
- Batch size: 8
- Optimizer: AdamW
- Learning rate: `2e-5`
- Device: MPS
- Training batches: 1,444
- Average training loss: `1.0173`

### Results

- Accuracy: `64.37%`
- Macro F1: `0.64`
- Weighted F1: `0.62`

### Classification Report

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| 0 | 0.69 | 0.86 | 0.77 | 633 |
| 1 | 0.54 | 0.75 | 0.63 | 299 |
| 2 | 0.60 | 0.65 | 0.63 | 385 |
| 3 | 0.67 | 0.81 | 0.74 | 610 |
| 4 | 0.65 | 0.36 | 0.46 | 961 |

### Confusion Matrix

| Actual / Predicted | 0 | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|---:|
| **0** | 546 | 17 | 18 | 7 | 45 |
| **1** | 38 | 225 | 8 | 7 | 21 |
| **2** | 33 | 10 | 251 | 27 | 64 |
| **3** | 18 | 15 | 24 | 495 | 58 |
| **4** | 156 | 148 | 115 | 200 | 342 |

### Observation

BERT achieved 64.37% accuracy after one epoch, compared with approximately 50.45% for the TF-IDF + LinearSVC baseline.

Class 4 remained the main source of errors, with a recall of 0.36 and an F1-score of 0.46.

---

## Experiment 2 — BERT, 3 Epochs

### Configuration

- Model: `bert-base-uncased`
- Epochs: 3
- Batch size: 8
- Optimizer: AdamW
- Learning rate: `2e-5`
- Device: MPS
- Training batches: 4,332
- Average training loss: `0.8401`

### Results

- Accuracy: `65.17%`
- Macro F1: `0.65`
- Weighted F1: `0.63`

### Classification Report

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| 0 | 0.67 | 0.89 | 0.77 | 633 |
| 1 | 0.57 | 0.65 | 0.61 | 299 |
| 2 | 0.61 | 0.64 | 0.62 | 385 |
| 3 | 0.67 | 0.85 | 0.75 | 610 |
| 4 | 0.68 | 0.38 | 0.49 | 961 |

### Confusion Matrix

| Actual / Predicted | 0 | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|---:|
| **0** | 562 | 13 | 15 | 9 | 34 |
| **1** | 49 | 194 | 5 | 10 | 41 |
| **2** | 43 | 12 | 245 | 29 | 56 |
| **3** | 18 | 8 | 25 | 517 | 42 |
| **4** | 164 | 113 | 113 | 207 | 364 |

### Observation

Increasing training from one to three epochs reduced the average training loss from `1.0173` to `0.8401`.

However, the improvement on the test set was relatively small:

- Accuracy: `64.37% → 65.17%`
- Macro F1: `0.64 → 0.65`
- Weighted F1: `0.62 → 0.63`

Class 4 remained the main source of errors. Its recall increased from `0.36` to `0.38`, while its F1-score increased from `0.46` to `0.49`.

The results suggest that simply increasing the number of fine-tuning epochs provides diminishing returns with the current BERT training configuration.

---

## Experiment 3 — BERT + Dynamic Padding + LR Scheduler

### Configuration

- Model: `bert-base-uncased`
- Epochs: 3
- Batch size: 8
- Optimizer: AdamW
- Initial learning rate: `2e-5`
- Learning rate scheduler: linear decay
- Warmup: 10% of training steps
- Dynamic padding: enabled
- Device: MPS
- Training batches: 4,332
- Average training loss: `0.8493`

### Results

- Accuracy: `63.12%`
- Macro F1: `0.63`
- Weighted F1: `0.62`

### Classification Report

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| 0 | 0.68 | 0.83 | 0.75 | 633 |
| 1 | 0.53 | 0.60 | 0.56 | 299 |
| 2 | 0.60 | 0.60 | 0.60 | 385 |
| 3 | 0.67 | 0.81 | 0.73 | 610 |
| 4 | 0.60 | 0.41 | 0.49 | 961 |

### Confusion Matrix

| Actual / Predicted | 0 | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|---:|
| **0** | 523 | 21 | 20 | 16 | 53 |
| **1** | 42 | 179 | 8 | 8 | 62 |
| **2** | 35 | 11 | 231 | 32 | 76 |
| **3** | 15 | 10 | 21 | 495 | 69 |
| **4** | 154 | 114 | 106 | 192 | 395 |

### Observation

Adding dynamic padding, a linear learning-rate scheduler, and a 10% warmup period did not improve overall test performance.

Compared with the three-epoch BERT configuration using a constant learning rate:

- Accuracy: `65.17% → 63.12%`
- Macro F1: `0.65 → 0.63`
- Weighted F1: `0.63 → 0.62`

Class 4 recall increased from `0.38` to `0.41`, but its F1-score remained at `0.49`. Performance decreased across several of the other classes, particularly classes 1 and 2.

The experiment suggests that the linear learning-rate schedule with 10% warmup does not provide an advantage for the current fine-tuning setup. The three-epoch BERT configuration with a constant `2e-5` learning rate remains the best-performing configuration tested so far.

---

## Experiment 4 — RoBERTa

_To be completed._