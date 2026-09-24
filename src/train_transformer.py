####### USEFUL IMPORT #######

# useful libs
import pandas as pd
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    DataCollatorWithPadding,  # to add dynamic padding
    )

# torch utilities
import torch
from torch.utils.data import Dataset

# for a better ui 
import time
import subprocess
import readchar 

# time import for better read
from datetime import datetime

##############################

# HYPERPARAMETERS SHORTCUT
# easier to modify 
model_str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext" # text for strings
MODEL_NAME = model_str # for execution model naming
learning_rate = 2e-5
num_epochs = 2
output_dir = ("models/biomedbert-medintake")


######### READING THE CSV FILES #########

print("\n" + "=" * 50)
print("MEDINTAKE")
print("=" * 50)

# train/test load
print("\nLoading the train/test data from clinical_cases public dataset ...")
df_train = pd.read_csv("datasets/clinical_cases_train.csv")
df_test = pd.read_csv("datasets/clinical_cases_test.csv")

# using sleep() to hault the code execution
time.sleep(6)
print("\nUpload completed succesfully.")

# shape of dfs
print(f"\nThe shape of train dataframe is: {df_train.shape}\n")
print(f"The shape of test dataframe is: {df_test.shape}")

########################################

print("\nPress any key to continue ...")
readchar.readkey()

# clearing the screen
subprocess.run(["clear"])

########################################






###### TOKENIZATION PHASE #########

print("\n" + "=" * 50)
print("MEDINTAKE")
print("=" * 50)

print("\nStarting tokenization process ...\n")
# tokenizer chosen automatically for ROBERTA
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

time.sleep(6)

# listing the dfs to make them compatible with the tokenizer 
X_train = df_train["medical_abstract"].tolist()
y_train = [label - 1 for label in df_train["condition_label"].tolist()]

from sklearn.model_selection import train_test_split 

X_train, X_val, y_train, y_val = train_test_split(
    X_train, 
    y_train,
    test_size=0.1,
    random_state=42,
    stratify=y_train
)

X_test = df_test["medical_abstract"].tolist()
y_test = [label - 1 for label in df_test["condition_label"].tolist()]

# tokenization of the entire dataset
train_encodings = tokenizer(
    X_train,    # all medical descriptions
    truncation=True,    # if text lenght is longer than what roberta requires it's cutted
    max_length=256
)

val_encodings = tokenizer(
    X_val,
    truncation=True,
    max_length=256
)

test_encodings = tokenizer(     # same goes here...
    X_test,
    truncation=True,
    max_length=256
)

# creating data collator with padding as hf recommend done at each batch creation 
data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)


# let's check what we obtained
print("\nType of the encoding created by the tokenizer: ")
print(type(train_encodings))    # type of the encoding created by the tokenizer


print(f"\nFirst {model_str} token IDs: ")
print(train_encodings["input_ids"][0][:20])     


print("\nAttention mask for each token showed above (1 = real token / 0 = padding): ")
print(train_encodings["attention_mask"][0][:20])    # mask shows how there is no padding yet
# print(train_encodings.keys()) shows all the masks 

#############################################

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
val_dataset = MedicalDataset(val_encodings, y_val)
test_dataset = MedicalDataset(test_encodings, y_test)


######################################

print("\nPress any key to continue ...")
readchar.readkey()

# clearing the screen
subprocess.run(["clear"])

#####################################









########### MEDICAL DATASET OVERVIEW ############

print("\n" + "=" * 50)
print("MEDINTAKE")
print("=" * 50)

print(f"\nTraining samples : {len(train_dataset)}")
print(f"Test samples     : {len(test_dataset)}")
print("Labels           : 5")
print(f"Model            : {model_str}")

time.sleep(6)

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


#########################################

print("\nPress any key to continue ...")
readchar.readkey()

# clearing the screen
subprocess.run(["clear"])

#########################################









########## DATA LOAD INTO BERT ###########
print("\n" + "=" * 50)
print("MEDINTAKE")
print("=" * 50)

from torch.utils.data import DataLoader


# DATA LOADERS #
train_loader = DataLoader(
    train_dataset,
    batch_size=8,   # 8 cases each batch
    shuffle=True,   # all the cases are mixed during the training process 
    collate_fn=data_collator # adding data collator
)

val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    collate_fn=data_collator
)

test_loader = DataLoader(
    test_dataset,   
    batch_size=8,
    collate_fn=data_collator
)   # not essential to shuffle on test 



batch = next(iter(train_loader))

print("\nDATALOADER")

print(f"Batch size       : {batch['input_ids'].shape[0]}")
print(f"Input shape      : {batch['input_ids'].shape}")
print(f"Attention shape  : {batch['attention_mask'].shape}")
print(f"Labels shape     : {batch['labels'].shape}")
print(f"Labels           : {batch['labels'].tolist()}")

print("\nChecking sequence lengths for 5 batches:") # check for better compute complexity 

for i, batch in enumerate(train_loader):
    print(f"Batch {i + 1}: {batch['input_ids'].shape}")

    if i == 4:
        break

#########################################

print("\nPress any key to continue ...")
readchar.readkey()

# clearing the screen
subprocess.run(["clear"])

#########################################








###### BERT MODEL + CLASSIFIER ######

print("\n" + "=" * 50)
print("MEDINTAKE")
print("=" * 50)

print("\nWEIGHTS LOAD ... \n")

from transformers import AutoModelForSequenceClassification 

# hugging face adds a classification head on bert 

model = AutoModelForSequenceClassification.from_pretrained( # bert model 
    MODEL_NAME,    # we use pre-train weights
    num_labels = 5      # our classification problem has 5 labels 
)


# using sleep() to hault the code execution
time.sleep(6)

print("\n" + "=" * 50)
print("MODEL")
print("=" * 50)

print(f"Model            : {model_str}")
print(f"Number of labels : {model.num_labels}")
print(f"Classifier       : {model.classifier}")


###### MPS DEVICE MACOS ##############

# macOS stuff
device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)
model.to(device) # put the model into chosen device, mps instead of cpu if available

print("\n" + "=" * 50)
print("DEVICE")
print("=" * 50)

print(f"Using           : {device}")

#########################################

print("\nPress any key to continue ...")
readchar.readkey()

# clearing the screen
subprocess.run(["clear"])

#########################################







######### SET ADAM OPZIMIZER ###########

from torch.optim import AdamW

optimizer = AdamW( # adamW optimizer 
    model.parameters(), # adam modifies BERT and classifier parameters
    lr=learning_rate     # starting learning rate
)

print("\n" + "=" * 50)
print("OPTIMIZER")
print("=" * 50)

print("\nOptimizer        : AdamW")
print(f"Learning rate    : {learning_rate}")

#########################################

print("\nPress any key to continue ...")
readchar.readkey()

# clearing the screen
subprocess.run(["clear"])

#########################################





print("\n" + "=" * 50)
print(f"TRAINING - {num_epochs} EPOCHS")
print("=" * 50)


########### TRAINING STEP / 3 EPOCHS ################

model.train()

losses = []

for epoch in range(num_epochs): # add 2 epochs 
    print(f"\nEPOCH {epoch + 1}/{num_epochs}")

    print(f"(started at {datetime.now().strftime('%H:%M:%S')})")

    epoch_losses = []

    for batch_idx, batch in enumerate(train_loader):

        batch = {
            key: value.to(device)
            for key, value in batch.items()
        }

        outputs = model(**batch)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_value = loss.item()

        losses.append(loss_value)
        epoch_losses.append(loss_value)

        if (batch_idx + 1) % 100 == 0:
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] "
                f"Batch {batch_idx + 1}/{len(train_loader)} "
                f"- Loss: {loss.item():.4f}"
            )

            # loss average of this epoch 
        average_epoch_loss = sum(epoch_losses) / len(epoch_losses)

    print(
        f"Epoch {epoch + 1} completed "
        f"- Average Loss: {average_epoch_loss:.4f}"
        )

    model.eval()

    val_losses = []

    with torch.no_grad():
        for batch in val_loader:
            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            outputs = model(**batch)
            val_losses.append(outputs.loss.item())

    average_val_loss = sum(val_losses) / len(val_losses)

    print(
        f"Validation loss: {average_val_loss:.4f}"
    )

    model.train()
    

############ TRAIN RESULTS ##################

average_loss = sum(losses) / len(losses)

print("\n" + "=" * 50)
print("EPOCH RESULTS")
print("=" * 50)

print(f"Average training loss : {average_loss:.4f}")
print(f"Batches              : {len(losses)}")

######### SAVE MODEL ####################

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

############## EVALUATION ##################

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

model.eval() # model in evaluation mode

all_predictions = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        batch = {key: value.to(device) for key, value in batch.items()}

        outputs = model(**batch)
        predictions = torch.argmax(outputs.logits, dim=1)

        all_predictions.extend(predictions.cpu().tolist())
        all_labels.extend(batch["labels"].cpu().tolist())

# accuracy 
print(f"Accuracy: {accuracy_score(all_labels, all_predictions):.4f}")
print(classification_report(all_labels, all_predictions))

# confusion matrix
cm = confusion_matrix(all_labels, all_predictions)
print("\nConfusion Matrix:")
print(cm)


############### GRAPHS ####################

# commented to not make it lag lol

#import matplotlib.pyplot as plt 

#plt.figure(figsize=(10, 5))
#plt.plot(losses)
#plt.xlabel("Batch")
#plt.ylabel("Loss")
#plt.title("Training loss")
#plt.grid(True)
#plt.show()

#########################################

print("\nPress any key to continue ...")
readchar.readkey()

# clearing the screen
subprocess.run(["clear"])

#########################################

  