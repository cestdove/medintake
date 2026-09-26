# usual libs
import pandas as pd
import torch 

# useful import
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# for a better ui 
import time
import subprocess
import readchar 





############ CSV AND MODELS LOAD ##############

print("\n" + "=" * 50)
print("MEDINTAKE")
print("=" * 50)

df_test = pd.read_csv("datasets/clinical_cases_test.csv")

tokenizer = AutoTokenizer.from_pretrained( # tokenizer from our model 
    "models/bert-medintake"
)

model = AutoModelForSequenceClassification.from_pretrained( # model from our model :D
    "models/bert-medintake"
)

device = torch.device( # if possible mps as device else cpu 
    "mps" if torch.backends.mps.is_available() else "cpu"
)

model.to(device) # load model into device as defined 



X_test = df_test["medical_abstract"].tolist() # defining df input 

y_test = [  # defining df target 
    label - 1
    for label in df_test["condition_label"].tolist()
]

test_encodings = tokenizer( # defining encodings to be created by tokenizer 
    X_test,
    truncation=True,
    padding=True
)


# defining class to call elements by index from the medical dataset 
class MedicalDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(val[idx])
            for key, val in self.encodings.items()
        }

        item["labels"] = torch.tensor(self.labels[idx])

        return item



test_dataset = MedicalDataset(test_encodings, y_test) # setting dataset by passing data to class

# dataloader definition to pass it for training to the model 
test_loader = DataLoader(
    test_dataset, # use test dataset defined by class before 
    batch_size=8 # use 8 abstracts as batch size each forward-backward pass
)

###################################################





############### VISUAL DATA CHECK #################

# little check 
print(f"Test samples : {len(test_dataset)}")
print(f"Batch size   : 8")
print(f"Batches      : {len(test_loader)}")

#########################################

print("\nPress any key to continue ...")
readchar.readkey()

# clearing the screen
subprocess.run(["clear"])

#########################################





############### MODEL EVALUATION #############

print("\n" + "=" * 50)
print("MEDINTAKE — EVALUATION")
print("=" * 50)

model.eval() # put model in eval mode

# lists to contain results
all_predictions = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        batch = {
            key: value.to(device)
            for key, value in batch.items()
        }

        outputs = model(**batch)

        predictions = torch.argmax(outputs.logits, dim=1)

        all_predictions.extend(predictions.cpu().tolist())
        all_labels.extend(batch["labels"].cpu().tolist())


# accuracy score
accuracy = accuracy_score(all_labels, all_predictions)

print(f"\nAccuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(all_labels, all_predictions))

# confusion matrix
cm = confusion_matrix(all_labels, all_predictions)
print("\nConfusion Matrix:")
print(cm)




