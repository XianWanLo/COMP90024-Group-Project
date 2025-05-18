import json
# import random
import numpy as np
from typing import Dict, List, Tuple
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_recall_fscore_support
import faiss

# === Parameters ===
TRAIN_CLAIMS_PATH = "data/train-claims.json"
DEV_CLAIMS_PATH = "data/dev-claims.json"
EVIDENCE_PATH = "data/evidence.json"
FINETUNED_MODEL_PATH = "models/bge-finetuned"
BASE_MODEL_NAME = "BAAI/bge-base-en-v1.5"
PREDICTIONS_SAVE_PATH = "MyPredictions"

# === Parameters ===
BATCH_SIZE = 16
TOP_K = 5 # Min 1 evidence and max 5 evidences
SIMILARITY_THRESHOLD = 0.90 # Used for selecting evidences

# === Loaders ===
def load_claims(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)

def load_evidences(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)

def predictions(cand_map: Dict[str, List[str]]) -> Dict[str, dict]:
    return {cid: {"evidences": eids} for cid, eids in cand_map.items()} # convert to the expected predictions structure

def evaluate(pred: dict, actual: dict):
    # For each claim, get the set of gold and predicted evidence IDs
    gold_sets = [set(actual[c]["evidences"]) for c in actual]
    pred_sets = [set(pred.get(c, {}).get("evidences", [])) for c in actual]

    # Fit the label binarizer on the gold evidence universe
    mlb = MultiLabelBinarizer()
    y_true = mlb.fit_transform(gold_sets)

    valid_labels = set(mlb.classes_)  # only evidence IDs that appear in gold

    # Filter predictions to only include valid evidence IDs (safe for shared use)
    pred_sets_filtered = [
        [eid for eid in preds if eid in valid_labels]
        for preds in pred_sets
    ]
    y_pred = mlb.transform(pred_sets_filtered)

    # Micro-averaged precision/recall/F1 (shared evidences per-claim are handled naturally)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )

    return rec, prec, f1

def faiss_candidates(
    claims: Dict[str, dict], evidences: Dict[str, str],
    model: SentenceTransformer, top_k: int
) -> Tuple[List[str], List[str], Dict[str, List[str]]]:
    claim_ids = list(claims)
    claim_texts = [claims[cid]["claim_text"] for cid in claim_ids]
    evidence_ids, evidence_texts = zip(*[(eid, txt) for eid, txt in evidences.items() if txt]) if evidences else ([], [])
  
    emb_e = model.encode(evidence_texts, batch_size=64, show_progress_bar=True,
                            convert_to_numpy=True, normalize_embeddings=True, device='cpu')

    emb_c = model.encode(claim_texts, batch_size=64, show_progress_bar=True,
                         convert_to_numpy=True, normalize_embeddings=True, device='cpu')

    d = emb_e.shape[1]
    index = faiss.IndexHNSWFlat(d, 32) 
    index.hnsw.efConstruction = 200

    print("Adding embeddings to FAISS index in batches")

    total = len(emb_e)
    progress_checkpoints = {int(total * i / 10) for i in range(1, 11)}  # 10%, 20%, ..., 100%

    for i in range(0, total, 100):
        try:
            batch = emb_e[i:i+100]
            index.add(batch)
            current = i + len(batch)
            if current in progress_checkpoints:
                print(f"{int(current / total * 100)}% complete ({current} / {total})", flush=True)
        except Exception as e:
            print(f"Failed at batch {i}: {e}")
            break

    print(f"Finished adding all vectors")

    print("Performing FAISS search (per claim)")

    I_all = []

    for emb in tqdm(emb_c, desc="Searching claims"):
        sim_scores, I = index.search(np.expand_dims(emb, axis=0), top_k)
        paired = list(zip(I[0], sim_scores[0]))
        
        # Filter by similarity threshold
        filtered = [idx for idx, score in paired if score >= SIMILARITY_THRESHOLD]

        # If nothing passes threshold, take the best one as we want at-least 1 evidence per claim
        if not filtered:
            filtered = [paired[0][0]] if paired else []
        
        I_all.append(filtered)

    print(f"Search done")

    # Create dict structure for the results
    cand_map: Dict[str, List[str]] = {
        claim_ids[i]: [
            evidence_ids[j] for j in I_all[i] if j < len(evidence_ids)
        ] for i in range(len(claim_ids))
    }

    return cand_map

# load model
model = SentenceTransformer(FINETUNED_MODEL_PATH)

dev_claims = load_claims(DEV_CLAIMS_PATH)
evidences = load_evidences(EVIDENCE_PATH)

cand_map = faiss_candidates(dev_claims, evidences, model, TOP_K)
pred = predictions(cand_map) # Convert to appropriate predictions structure

# save predictions file
if PREDICTIONS_SAVE_PATH:
    with open(PREDICTIONS_SAVE_PATH, "w") as f:
        json.dump(pred, f, indent=2)
    print(f"Saved predictions to {PREDICTIONS_SAVE_PATH}")

recall, precision, f1 = evaluate(pred, dev_claims) # evaluate results
print(f"Recall: {recall:.4f}, Precision: {precision:.4f}, F1: {f1:.4f}")

'''
Batches: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 18888/18888 [6:02:03<00:00,  4.86it/s]
Batches: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:01<00:00,  2.04it/s]
Adding embeddings to FAISS index in batches
10% complete (120882 / 1208827)
20% complete (241765 / 1208827)
30% complete (362648 / 1208827)
40% complete (483530 / 1208827)
50% complete (604413 / 1208827)
60% complete (725296 / 1208827)
70% complete (846179 / 1208827)
80% complete (967062 / 1208827)
90% complete (1087945 / 1208827)
100% complete (1208827 / 1208827)
Finished adding all vectors
Performing FAISS search (per claim)
Searching claims: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 154/154 [00:56<00:00,  2.75it/s]
Search done 
Saved predictions to MyPredictions
Recall: 0.0896, Precision: 0.1800, F1: 0.1176
'''
