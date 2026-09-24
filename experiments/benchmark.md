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
| RoBERTa | `roberta-base` | 3 | 8 | `2e-5` | 63.61% | 0.63 | 0.61 |

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
- Epochs: 3
- Batch size: 8
- Optimizer: AdamW
- Learning rate: `2e-5`
- Device: MPS
- Dynamic padding: enabled
- Learning-rate scheduler: none
- Training batches per epoch: 1,444
- Total training batches: 4,332
- Average training loss: `0.8422`

## Results

- Accuracy: **63.61%**
- Macro F1: **0.63**
- Weighted F1: **0.61**

## Classification Report

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| 0 | 0.70 | 0.83 | 0.76 | 633 |
| 1 | 0.53 | 0.76 | 0.62 | 299 |
| 2 | 0.59 | 0.68 | 0.63 | 385 |
| 3 | 0.61 | 0.90 | 0.73 | 610 |
| 4 | 0.75 | 0.28 | 0.41 | 961 |

## Confusion Matrix

| Actual / Predicted | 0 | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|---:|
| **0** | 527 | 26 | 34 | 24 | 22 |
| **1** | 38 | 227 | 5 | 11 | 18 |
| **2** | 22 | 11 | 263 | 56 | 33 |
| **3** | 12 | 16 | 14 | 548 | 20 |
| **4** | 159 | 149 | 128 | 253 | 272 |

## Observation

RoBERTa achieved **63.61% accuracy**, with a Macro F1 of **0.63** and Weighted F1 of **0.61**.

Compared with the three-epoch BERT configuration using a constant learning rate:

| Metric | BERT 3 Epochs | RoBERTa 3 Epochs | Change |
|---|---:|---:|---:|
| Accuracy | **65.17%** | 63.61% | -1.56 pp |
| Macro F1 | **0.65** | 0.63 | -0.02 |
| Weighted F1 | **0.63** | 0.61 | -0.02 |

RoBERTa also required substantially longer training time on the current MPS hardware. The complete three-epoch run took approximately eight hours.

Class 4 remained the main source of errors, with a recall of `0.28` and an F1-score of `0.41`.

The results show that, under the tested configuration, RoBERTa did not improve over the BERT baseline despite completing the full three-epoch training run.

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

RoBERTa completed the full three-epoch experiment but achieved lower test performance than BERT:

- Accuracy: `63.61%`
- Macro F1: `0.63`
- Weighted F1: `0.61`

RoBERTa also required substantially more training time on the current hardware.

Across the completed Transformer experiments, **class 4 consistently represents the most difficult class**, particularly in terms of recall.

---

# Experiment 5 — RoBERTa + Validation Split

## Configuration

- Model: `roberta-base`
- Epochs: 3
- Batch size: 8
- Optimizer: AdamW
- Learning rate: `2e-5`
- Maximum sequence length: `256`
- Validation split: 10%
- Split strategy: stratified
- Random state: 42
- Device: MPS
- Dynamic padding: enabled
- Learning-rate scheduler: none
- Training batches per epoch: 1,300
- Total training batches: 3,900

## Results

- Average training loss: `0.8763`
- Validation loss:
  - Epoch 1: `0.8596`
  - Epoch 2: `0.8638`
  - Epoch 3: `0.8423`
- Test Accuracy: **`61.98%`**
- Test Macro F1: **`0.62`**
- Test Weighted F1: **`0.61`**

## Classification Report

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| 0 | 0.70 | 0.80 | 0.74 | 633 |
| 1 | 0.56 | 0.51 | 0.54 | 299 |
| 2 | 0.53 | 0.77 | 0.63 | 385 |
| 3 | 0.70 | 0.69 | 0.69 | 610 |
| 4 | 0.56 | 0.44 | 0.49 | 961 |

## Confusion Matrix

| Actual / Predicted | 0 | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|---:|
| **0** | 504 | 17 | 52 | 10 | 50 |
| **1** | 37 | 153 | 8 | 8 | 93 |
| **2** | 20 | 7 | 295 | 11 | 52 |
| **3** | 13 | 6 | 43 | 419 | 129 |
| **4** | 151 | 88 | 155 | 148 | 419 |

## Observation

This experiment introduces a stratified 10% validation split from the original training set while keeping the test set unchanged.

The training loss decreased across all three epochs:

| Epoch | Training Loss | Validation Loss |
|---|---:|---:|
| 1 | 1.0185 | 0.8596 |
| 2 | 0.8404 | 0.8638 |
| 3 | 0.7699 | **0.8423** |

Validation loss was lowest after the third epoch. The results do not show a clear increase in validation loss across the three epochs.

Because the validation split reduces the amount of data available for training, the resulting test performance should not be treated as a direct replacement for the previous three-epoch RoBERTa experiment.

The experiment establishes a validation-based training setup for subsequent hyperparameter experiments.

---

# Experiment 6 — RoBERTa + Learning Rate 1e-5

## Configuration

- Model: `roberta-base`
- Epochs: 3
- Batch size: 8
- Optimizer: AdamW
- Learning rate: `1e-5`
- Maximum sequence length: `256`
- Validation split: 10%
- Split strategy: stratified
- Random state: 42
- Device: MPS
- Dynamic padding: enabled
- Learning-rate scheduler: none
- Training batches per epoch: 1,300
- Total training batches: 3,900

The configuration is identical to Experiment 5 except for the learning rate, which was reduced from `2e-5` to `1e-5`.

## Results

- Average training loss: `0.8435`
- Validation loss:
  - Epoch 1: `0.8753`
  - Epoch 2: `0.8225`
  - Epoch 3: `0.8400`
- Test Accuracy: **`62.98%`**
- Test Macro F1: **`0.63`**
- Test Weighted F1: **`0.61`**

## Classification Report

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| 0 | 0.67 | 0.83 | 0.74 | 633 |
| 1 | 0.53 | 0.67 | 0.59 | 299 |
| 2 | 0.58 | 0.65 | 0.61 | 385 |
| 3 | 0.66 | 0.82 | 0.73 | 610 |
| 4 | 0.64 | 0.36 | 0.46 | 961 |

## Confusion Matrix

| Actual / Predicted | 0 | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|---:|
| **0** | 524 | 30 | 32 | 16 | 31 |
| **1** | 36 | 200 | 8 | 5 | 50 |
| **2** | 32 | 10 | 250 | 34 | 59 |
| **3** | 16 | 9 | 27 | 501 | 57 |
| **4** | 171 | 131 | 117 | 198 | 344 |

## Observation

Reducing the learning rate from `2e-5` to `1e-5` improved test accuracy from `61.98%` to `62.98%` under the validation-based RoBERTa setup.

| Metric | LR `2e-5` | LR `1e-5` | Change |
|---|---:|---:|---:|
| Accuracy | 61.98% | **62.98%** | +1.00 pp |
| Macro F1 | 0.62 | **0.63** | +0.01 |
| Weighted F1 | 0.61 | 0.61 | 0.00 |

Validation loss reached its minimum after the second epoch (`0.8225`) and increased slightly by the third epoch (`0.8400`), while training loss continued to decrease.

This provides an initial indication that the third epoch may introduce some overfitting under the `1e-5` learning rate, although the difference in validation loss is small.

Class 4 remained the most difficult class, with a recall of `0.36` and an F1-score of `0.46`.

The experiment supports continuing hyperparameter development with the `1e-5` learning rate while keeping the remaining configuration fixed.

---

# Experiment 7 — RoBERTa + 2 Epochs

## Configuration

- Model: `roberta-base`
- Epochs: 2
- Batch size: 8
- Optimizer: AdamW
- Learning rate: `1e-5`
- Maximum sequence length: `256`
- Validation split: 10%
- Split strategy: stratified
- Random state: 42
- Device: MPS
- Dynamic padding: enabled
- Learning-rate scheduler: none
- Training batches per epoch: 1,300
- Total training batches: 2,600

The configuration is identical to Experiment 6 except for the number of training epochs, which was reduced from 3 to 2.

## Results

- Average training loss: `0.9164`
- Validation loss:
  - Epoch 1: `0.8556`
  - Epoch 2: `0.8441`
- Test Accuracy: **`63.99%`**
- Test Macro F1: **`0.64`**
- Test Weighted F1: **`0.63`**

## Classification Report

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| 0 | 0.66 | 0.91 | 0.76 | 633 |
| 1 | 0.57 | 0.64 | 0.60 | 299 |
| 2 | 0.64 | 0.57 | 0.60 | 385 |
| 3 | 0.69 | 0.76 | 0.72 | 610 |
| 4 | 0.60 | 0.42 | 0.49 | 961 |

## Confusion Matrix

| Actual / Predicted | 0 | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|---:|
| **0** | 575 | 8 | 11 | 9 | 30 |
| **1** | 51 | 192 | 3 | 2 | 51 |
| **2** | 41 | 10 | 218 | 27 | 89 |
| **3** | 18 | 12 | 22 | 463 | 95 |
| **4** | 189 | 116 | 88 | 168 | 400 |

## Observation

Reducing the number of training epochs from 3 to 2 while keeping the learning rate at `1e-5` improved test performance under the validation-based RoBERTa setup.

| Metric | 3 Epochs | 2 Epochs | Change |
|---|---:|---:|---:|
| Accuracy | 62.98% | **63.99%** | +1.01 pp |
| Macro F1 | 0.63 | **0.64** | +0.01 |
| Weighted F1 | 0.61 | **0.63** | +0.02 |

Validation loss decreased from `0.8556` after the first epoch to `0.8441` after the second epoch.

Class 4 also showed improved recall compared with the three-epoch configuration, increasing from `0.36` to `0.42`, while its F1-score increased from `0.46` to `0.49`.

The results indicate that, within this validation-based configuration, two epochs with a `1e-5` learning rate provided better test performance than three epochs.

# Experiment 8 — BERT + Validation Split

---

## Configuration

- Model: `bert-base-uncased`
- Epochs: 2
- Batch size: 8
- Optimizer: AdamW
- Learning rate: `1e-5`
- Maximum sequence length: `256`
- Validation split: 10%
- Split strategy: stratified
- Random state: 42
- Device: MPS
- Dynamic padding: enabled
- Learning-rate scheduler: none
- Training batches per epoch: 1,300
- Total training batches: 2,600

The configuration follows the same validation-based protocol used in Experiments 5–7. The learning rate and number of epochs match Experiment 7, while the Transformer architecture is changed from RoBERTa to BERT.

## Results

- Average training loss: `0.9531`
- Validation loss:
  - Epoch 1: `0.8793`
  - Epoch 2: `0.8692`
- Test Accuracy: **`64.34%`**
- Test Macro F1: **`0.64`**
- Test Weighted F1: **`0.63`**

## Classification Report

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| 0 | 0.66 | 0.91 | 0.76 | 633 |
| 1 | 0.58 | 0.67 | 0.62 | 299 |
| 2 | 0.62 | 0.61 | 0.61 | 385 |
| 3 | 0.68 | 0.79 | 0.73 | 610 |
| 4 | 0.63 | 0.38 | 0.48 | 961 |

## Confusion Matrix

| Actual / Predicted | 0 | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|---:|
| **0** | 574 | 13 | 13 | 8 | 25 |
| **1** | 47 | 200 | 4 | 5 | 43 |
| **2** | 45 | 9 | 235 | 27 | 69 |
| **3** | 17 | 8 | 25 | 480 | 80 |
| **4** | 187 | 117 | 103 | 185 | 369 |

## Observation

BERT achieved **64.34% test accuracy**, with a Macro F1 of **0.64** and Weighted F1 of **0.63** under the validation-based training protocol.

Validation loss decreased from `0.8793` after the first epoch to `0.8692` after the second epoch, while training loss decreased from `1.0698` to `0.8364`.

Compared with the matched two-epoch RoBERTa configuration from Experiment 7:

| Metric | BERT | RoBERTa | Difference |
|---|---:|---:|---:|
| Accuracy | **64.34%** | 63.99% | **+0.35 pp** |
| Macro F1 | 0.64 | 0.64 | 0.00 |
| Weighted F1 | 0.63 | 0.63 | 0.00 |

The two models produced the same Macro F1 and Weighted F1 at the reported precision, while BERT achieved a 0.35 percentage-point higher test accuracy.

Class 4 remained the most difficult class, with a recall of `0.38` and an F1-score of `0.48`.

The results provide a controlled comparison between BERT and RoBERTa under the same validation-based training configuration.

---

# Experiment 9 — BERT + 1e-5 + 3 Epochs

## Configuration

- Model: `bert-base-uncased`
- Epochs: 3
- Batch size: 8
- Optimizer: AdamW
- Learning rate: `1e-5`
- Maximum sequence length: `256`
- Validation split: 10%
- Split strategy: stratified
- Random state: 42
- Device: MPS
- Dynamic padding: enabled
- Learning-rate scheduler: none
- Training batches per epoch: 1,300
- Total training batches: 3,900

The configuration follows the same validation-based training protocol used in Experiments 5–8. Compared with Experiment 8, the learning rate remains at `1e-5`, while the number of training epochs is increased from 2 to 3.

## Results

- Average training loss: `0.8645`
- Validation loss:
  - Epoch 1: `0.8996`
  - Epoch 2: `0.8681`
  - Epoch 3: `0.8705`
- Test Accuracy: **`61.39%`**
- Test Macro F1: **`0.61`**
- Test Weighted F1: **`0.61`**

## Classification Report

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| 0 | 0.67 | 0.80 | 0.73 | 633 |
| 1 | 0.54 | 0.58 | 0.56 | 299 |
| 2 | 0.65 | 0.49 | 0.56 | 385 |
| 3 | 0.67 | 0.68 | 0.68 | 610 |
| 4 | 0.54 | 0.51 | 0.53 | 961 |

## Confusion Matrix

| Actual / Predicted | 0 | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|---:|
| **0** | 507 | 18 | 20 | 16 | 72 |
| **1** | 42 | 173 | 2 | 8 | 74 |
| **2** | 40 | 11 | 187 | 38 | 109 |
| **3** | 17 | 9 | 10 | 416 | 158 |
| **4** | 154 | 109 | 68 | 140 | 490 |

## Observation

BERT achieved **61.39% test accuracy**, with a Macro F1 of **0.61** and Weighted F1 of **0.61**.

Training loss continued to decrease across all three epochs:

- Epoch 1: `1.0403`
- Epoch 2: `0.8193`
- Epoch 3: `0.7340`

Validation loss decreased from `0.8996` to `0.8681` after the second epoch, but slightly increased to `0.8705` after the third epoch.

Compared with Experiment 8, which used the same learning rate and validation protocol but trained for only two epochs:

| Metric | 2 Epochs | 3 Epochs | Difference |
|---|---:|---:|---:|
| Accuracy | **64.34%** | 61.39% | **-2.95 pp** |
| Macro F1 | **0.64** | 0.61 | -0.03 |
| Weighted F1 | **0.63** | 0.61 | -0.02 |

The additional epoch therefore did not improve performance. The best validation loss was also reached after the second epoch, suggesting that extending training to three epochs was not beneficial for this configuration.

Class 4 remained difficult, although its recall increased from `0.38` in Experiment 8 to `0.51` in this experiment. This improvement did not translate into higher overall performance, as performance on other classes decreased.

The results indicate that increasing training from 2 to 3 epochs at a learning rate of `1e-5` is not an effective direction under the current configuration.

---

# Experiment 10 — BERT + 2e-5 + 2 Epochs

## Configuration

- Model: `bert-base-uncased`
- Epochs: 2
- Batch size: 8
- Optimizer: AdamW
- Learning rate: `2e-5`
- Maximum sequence length: `256`
- Validation split: 10%
- Split strategy: stratified
- Random state: 42
- Device: MPS
- Dynamic padding: enabled
- Learning-rate scheduler: none
- Training batches per epoch: 1,300
- Total training batches: 2,600

The configuration follows the same validation-based training protocol used in Experiments 8 and 9. Compared with Experiment 9, the learning rate is increased from `1e-5` to `2e-5`, while the number of epochs is reduced from 3 to 2.

## Results

- Average training loss: `0.9233`
- Validation loss:
  - Epoch 1: `0.8954`
  - Epoch 2: `0.8506`
- Test Accuracy: **`64.58%`**
- Test Macro F1: **`0.64`**
- Test Weighted F1: **`0.62`**

## Classification Report

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| 0 | 0.68 | 0.87 | 0.76 | 633 |
| 1 | 0.55 | 0.73 | 0.62 | 299 |
| 2 | 0.57 | 0.74 | 0.65 | 385 |
| 3 | 0.67 | 0.83 | 0.74 | 610 |
| 4 | 0.72 | 0.32 | 0.44 | 961 |

## Confusion Matrix

| Actual / Predicted | 0 | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|---:|
| **0** | 550 | 16 | 25 | 12 | 30 |
| **1** | 43 | 217 | 7 | 5 | 27 |
| **2** | 34 | 7 | 286 | 28 | 30 |
| **3** | 17 | 18 | 33 | 509 | 33 |
| **4** | 163 | 138 | 148 | 209 | 303 |

## Observation

BERT achieved **64.58% test accuracy**, with a Macro F1 of **0.64** and Weighted F1 of **0.62** under the validation-based training protocol.

Training loss decreased from `1.0219` after the first epoch to `0.8247` after the second epoch. Validation loss also decreased from `0.8954` to `0.8506`.

Compared with Experiment 9, which used the same validation protocol but a learning rate of `1e-5` and three epochs:

| Metric | Experiment 9 | Experiment 10 | Difference |
|---|---:|---:|---:|
| Accuracy | 61.39% | **64.58%** | **+3.19 pp** |
| Macro F1 | 0.61 | **0.64** | +0.03 |
| Weighted F1 | 0.61 | **0.62** | +0.01 |

The higher learning rate combined with two epochs substantially improved test accuracy compared with the `1e-5` three-epoch configuration.

Validation loss reached its lowest value after the second epoch, with no increase across the two training epochs.

Class 4 remained the most difficult class in terms of recall, with a recall of `0.32` and an F1-score of `0.44`. However, its precision increased to `0.72`, indicating that the model made relatively few incorrect predictions of class 4 while missing a substantial portion of its actual examples.

The configuration improved over Experiment 9 but remained below the historical benchmark of `65.17%`.

---

# Experiment 11 — BERT + 2e-5 + 3 Epochs

## Configuration

- Model: `bert-base-uncased`
- Epochs: 3
- Batch size: 8
- Optimizer: AdamW
- Learning rate: `2e-5`
- Maximum sequence length: `256`
- Validation split: 10%
- Split strategy: stratified
- Random state: 42
- Device: MPS
- Dynamic padding: enabled
- Learning-rate scheduler: none
- Training batches per epoch: 1,300
- Total training batches: 3,900

The configuration follows the same validation-based training protocol used in Experiments 8–10. Compared with Experiment 10, the learning rate remains at `2e-5`, while the number of training epochs is increased from 2 to 3.

## Results

- Average training loss: `0.8400`
- Validation loss:
  - Epoch 1: `0.8658`
  - Epoch 2: `0.8392`
  - Epoch 3: `0.8806`
- Test Accuracy: **`62.47%`**
- Test Macro F1: **`0.62`**
- Test Weighted F1: **`0.61`**

## Classification Report

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| 0 | 0.66 | 0.83 | 0.74 | 633 |
| 1 | 0.56 | 0.58 | 0.57 | 299 |
| 2 | 0.60 | 0.52 | 0.55 | 385 |
| 3 | 0.65 | 0.82 | 0.73 | 610 |
| 4 | 0.60 | 0.42 | 0.49 | 961 |

## Confusion Matrix

| Actual / Predicted | 0 | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|---:|
| **0** | 526 | 18 | 33 | 15 | 41 |
| **1** | 42 | 172 | 3 | 7 | 75 |
| **2** | 34 | 14 | 200 | 38 | 99 |
| **3** | 20 | 9 | 22 | 503 | 56 |
| **4** | 170 | 96 | 78 | 214 | 403 |

## Observation

BERT achieved **62.47% test accuracy**, with a Macro F1 of **0.62** and Weighted F1 of **0.61** under the validation-based training protocol.

Training loss continued to decrease across all three epochs:

- Epoch 1: `1.0039`
- Epoch 2: `0.8064`
- Epoch 3: `0.7097`

Validation loss decreased from `0.8658` after the first epoch to `0.8392` after the second epoch, but increased to `0.8806` after the third epoch.

Compared with Experiment 10, which used the same learning rate and validation protocol but trained for only two epochs:

| Metric | 2 Epochs | 3 Epochs | Difference |
|---|---:|---:|---:|
| Accuracy | **64.58%** | 62.47% | **-2.11 pp** |
| Macro F1 | **0.64** | 0.62 | -0.02 |
| Weighted F1 | **0.62** | 0.61 | -0.01 |

The additional epoch therefore reduced test performance. The validation loss reached its minimum after the second epoch and increased substantially during the third epoch while training loss continued to decrease, providing a clearer indication of overfitting than in the previous experiments.

Class 4 remained one of the main sources of errors, with a recall of `0.42` and an F1-score of `0.49`.

The results indicate that, under the current validation-based configuration, two epochs with a `2e-5` learning rate performed better than three epochs.

---

## Next Experiments

Planned directions include:

- Benchmarking more recent Transformer architectures.
- Evaluating biomedical-domain pretrained language models.
- Performing deeper Transformer error analysis.
- Comparing general-domain and biomedical-domain representations.
- Exploring retrieval-based approaches and Retrieval-Augmented Generation (RAG).
- Evaluating retrieval quality separately from generation quality.

---