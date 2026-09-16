# useful libs
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# torch utilities
import torch
from torch.utils.data import Dataset


# train/test load
df_train = pd.read_csv("datasets/clinical_cases_train.csv")
df_test = pd.read_csv("datasets/clinical_cases_test.csv")

# shape of dfs
print(df_train.shape)
print(df_test.shape)

# tokenizer chosen automatically for BERT
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# listing the dfs to make them compatible with the tokenizer 
X_train = df_train["medical_abstract"].tolist()
y_train = df_train["condition_label"].tolist()

X_test = df_test["medical_abstract"].tolist()
y_test = df_test["condition_label"].tolist()

# tokenization of the entire dataset
train_encodings = tokenizer(
    X_train,    # all medical descriptions
    truncation=True,    # if text lenght is longer than what BERT requires it's cutted
    padding=True    # shorter texts are filled with padding to the required lenght
)

test_encodings = tokenizer(     # same goes here...
    X_test,
    truncation=True,
    padding=True
)

# let's check what we obtained
print(type(train_encodings))    # type of the encoding created by the tokenizer
print(train_encodings["input_ids"][0][:20])     # BERT token IDs where 101 is [CLS]
print(train_encodings["attention_mask"][0][:20])    # mask shows how there is no padding yet
# print(train_encodings.keys()) shows all the masks 




