import os
import json
from typing import Dict
from tqdm import tqdm
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses

# === Parameters ===
TRAIN_CLAIMS_PATH = "data/train-claims.json"
EVIDENCE_PATH = "data/evidence.json"
FINETUNED_MODEL_PATH = "models/bge-finetuned"
BASE_MODEL_NAME = "BAAI/bge-base-en-v1.5"

EPOCHS = 1
BATCH_SIZE = 16

def load_claims(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)

def load_evidences(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)
    
def train_model(model: SentenceTransformer, dataloader, loss_func, output_path: str, epochs: int = 1):
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        model.fit(
            train_objectives=[(dataloader, loss_func)],
            epochs=1,
            warmup_steps=100,
            output_path=output_path,
            show_progress_bar=True
        )

        model.save(output_path)
        print(f"Finished training Epoch {epoch + 1}")

# Prepare training data 
print("Loading training data")
claims = load_claims(TRAIN_CLAIMS_PATH)

print("Loading evidences data")
evidences = load_evidences(EVIDENCE_PATH)
evidence_ids = list(evidences.keys())

print("Loading model.")
if os.path.exists(FINETUNED_MODEL_PATH):
    print(f"Continuing from: {FINETUNED_MODEL_PATH}")
    model = SentenceTransformer(FINETUNED_MODEL_PATH)
else:
    print(f"Starting from base model: {BASE_MODEL_NAME}")
    model = SentenceTransformer(BASE_MODEL_NAME)

# Get training examples 
train_examples = []
for cid, cdata in tqdm(claims.items(), desc="Building positive training samples"):
    claim_text = cdata["claim_text"]
    positive_eids = cdata["evidences"]
    positive_texts = [evidences[eid] for eid in positive_eids if eid in evidences]
    if not positive_texts:
        continue
    for pos_text in positive_texts:
        train_examples.append(InputExample(texts=[claim_text, pos_text]))

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)
train_loss = losses.MultipleNegativesRankingLoss(model)

try:
    train_model(model, train_dataloader, train_loss, FINETUNED_MODEL_PATH, epochs=EPOCHS)
    model.save(FINETUNED_MODEL_PATH)
    print(f"\nModel saved to: {FINETUNED_MODEL_PATH}")
except Exception as e:
    print(f"Error during training: {e}") 

'''
Loading training data
Loading evidences data
Loading model.
Continuing from: models/bge-finetuned
Building positive training samples: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1228/1228 [00:00<00:00, 47049.95it/s]

Epoch 1/1
{'train_runtime': 101.0694, 'train_samples_per_second': 40.784, 'train_steps_per_second': 2.553, 'train_loss': 0.3702929592871851, 'epoch': 1.0}                                               
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 258/258 [01:41<00:00,  2.55it/s]
Finished training Epoch 1

Model saved to: models/bge-finetuned
'''
    
    