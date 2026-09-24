# MedIntake

Machine learning-assisted platform for patient intake, case categorization/routing, and clinical workflow support.

This project evolves [PW_Triage_ML](https://github.com/cestdove/PW_Triage_ML), my previous thesis project exploring Support Vector Machines for ticket triage and text classification.

## Dataset

The model takes a short clinical-case description as input and predicts one of five broad medical-condition categories.

The project uses the public *Medical Abstracts Text Classification Dataset*, available on Hugging Face and created by Tim Schopf, Daniel Braun, and Florian Matthes. The dataset consists of English medical abstracts rather than real patient tickets or clinical records.

| Label | Category | Train | Test |
|---:|---|---:|---:|
| 1 | Neoplasms | 2,530 | 633 |
| 2 | Digestive system diseases | 1,195 | 299 |
| 3 | Nervous system diseases | 1,540 | 385 |
| 4 | Cardiovascular diseases | 2,441 | 610 |
| 5 | General pathological conditions | 3,844 | 961 |

## Approach

The project explores the progression from traditional text classification to Transformer-based NLP models.

### Classical Baseline

The initial baseline uses:

- TF-IDF vectorization
- LinearSVC
- Confusion matrices and classification metrics
- Error analysis and feature analysis

### Transformer Models

The project then evaluates several Transformer architectures using Hugging Face Transformers and PyTorch.

- **BERT (`bert-base-uncased`)** — initial Transformer baseline
- **RoBERTa (`roberta-base`)** — alternative general-domain Transformer architecture
- **BiomedBERT** — biomedical-domain pretrained model

All Transformer models use a maximum sequence length of 256 tokens and are fine-tuned for five-class classification.

### Results

The main results obtained during development are:

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| TF-IDF + LinearSVC | 50.45% | ~0.50 | — |
| BERT | 65.17% | 0.65 | 0.63 |
| RoBERTa | 63.99% | 0.64 | 0.63 |
| **BiomedBERT** | **65.37%** | **0.65** | **0.64** |

BiomedBERT currently provides the strongest result under the controlled experimental protocol, with **65.37% test accuracy**.

Compared with the matched BERT configuration, BiomedBERT improves accuracy by **0.79 percentage points** and Weighted F1 by **0.02**.

The complete experiment history, including training configurations, validation losses, classification reports, and confusion matrices, is available in `experiments/benchmark.md`.

## Project Structure

```text
medintake/
├── datasets/
├── models/
├── notebooks/
├── src/
│   ├── download_dataset.py
│   ├── train_model.py
│   ├── train_transformer.py
│   └── evaluate_transformer.py
├── experiments/
│   └── benchmark.md
├── README.md
├── requirements.txt
└── LICENSE
```

## Setup

```bash
git clone https://github.com/cestdove/medintake.git
cd medintake

pip install -r requirements.txt
```

To download and prepare the dataset:

```bash
python src/download_dataset.py
```

To train the baseline:

```bash
python src/train_model.py
```

To train the Transformer model:

```bash
python src/train_transformer.py
```

To evaluate a saved Transformer model:

```bash
python src/evaluate_transformer.py
```

## License

This project is licensed under the [MIT License](LICENSE).