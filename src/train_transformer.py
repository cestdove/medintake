# useful libs
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# torch utilities
import torch
from torch.utils.data import Dataset



print("\n" + "=" * 50)
print("DATA LOAD")
print("=" * 50)

# train/test load
print("\nLoading the train/test data from clinical_cases public dataset ...")
df_train = pd.read_csv("datasets/clinical_cases_train.csv")
df_test = pd.read_csv("datasets/clinical_cases_test.csv")

print("\nUpload completed succesfully.")

# shape of dfs
print(f"\nThe shape of train dataframe is: {df_train.shape}\n")
print(f"The shape of test dataframe is: {df_test.shape}")



###### TOKENIZATION #########

print("\n" + "=" * 50)
print("\nStarting tokenization process ...\n")
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
print("\nType of the encoding created by the tokenizer: ")
print(type(train_encodings))    # type of the encoding created by the tokenizer

print("\nFirst BERT token IDs (101 = [CLS] / 102 = [SEP]): ")
print(train_encodings["input_ids"][0][:20])     # BERT token IDs where 101 is [CLS]

print("\nAttention mask for each token showed above (1 = real token / 0 = padding): ")
print(train_encodings["attention_mask"][0][:20])    # mask shows how there is no padding yet
# print(train_encodings.keys()) shows all the masks 


# creating a class for better gestion 
class MedicalDataset(Dataset):  # from torch.utils.data.Dataset
    def __init__ (self, encodings, labels):
        self.encodings = encodings  # data created by the tokenizer
        self.labels = labels    # correct classes 

    def __len__(self):  # lenght of the dataset, so how much examples there are in it
        return len(self.labels)

    def __getitem__ (self, idx):    # to get a particular item by index from the dataset
        item = {
            key: torch.tensor(val[idx]) # idx ins the index we pick
            for key, val in self.encodings.items()  # for each element of BatchEncoding we pick idx
        }   # relevant is to convert the data to tensors for compatibility 
        item["labels"] = torch.tensor(self.labels[idx]) # add the correct label 

        return item 

# little check of what we did here with class creation
print("\nClass {MedicalDataset} created: ")
print(MedicalDataset)

print("\n" + "=" * 50)

# asigning with MedicalDataset call
train_dataset = MedicalDataset(train_encodings, y_train)
test_dataset = MedicalDataset(test_encodings, y_test)

# lenght of train dataset after tokenization and class call
lenght_train_df = len(train_dataset)
print(f"\nLenght of train dataset: {lenght_train_df}")  


############ 
import time
import subprocess

print("\n" + "=" * 50)
# printing the start time
print("\nThe time of code execution begin is : ", time.ctime())

# using sleep() to hault the code execution
time.sleep(6)

# Clearing the Screen
subprocess.run(["clear"])

######## OVERVIEW #########

print("\n" + "=" * 50)
print("MEDINTAKE — BERT DATASET")
print("=" * 50)

print(f"Training samples : {len(train_dataset)}")
print(f"Test samples     : {len(test_dataset)}")
print("Labels           : 5")
print("Model            : bert-base-uncased")

sample = train_dataset[0]

print("\n" + "-" * 50)
print("EXAMPLE")
print("-" * 50)

print(f"Label            : {sample['labels'].item()}")
print(f"Tokenized length : {sample['attention_mask'].sum().item()}")

tokens = tokenizer.convert_ids_to_tokens(sample["input_ids"])
real_tokens = tokens[:sample["attention_mask"].sum().item()]

print("\nTokens:")
print(" ".join(real_tokens[:400]))

print("=" * 50)



