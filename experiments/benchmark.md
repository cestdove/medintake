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

---

## Experiment 1 — BERT

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

Class 4 remains the main source of errors, with a recall of 0.36 and an F1-score of 0.46.

---

## Experiment 2 — BERT, 3 Epochs

_To be completed._