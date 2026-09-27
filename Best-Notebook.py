# ============================================================
# BENGALI NLP SAFETY CHALLENGE 2026
# STRONG OOF ENSEMBLE (NO EXTERNAL DEPENDENCY NEEDED)
# ============================================================

import os
import gc
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from scipy.sparse import hstack

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW  # PyTorch-এর নিজস্ব AdamW ব্যবহার করা হয়েছে

from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup
)

# ============================================================
# 1. CONFIG
# ============================================================

SEED = 42
N_FOLDS = 3
MAX_LEN = 192
BATCH_SIZE = 16

BERT_EPOCHS = 3
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01

TFIDF_C = 2.0

DATA_DIR = "/kaggle/input/competitions/bengali-nlp-safety-challenge-2026"

TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")
VALID_FILE = os.path.join(DATA_DIR, "validation.csv")
TEST_FILE = os.path.join(DATA_DIR, "test.csv")
SAMPLE_FILE = os.path.join(DATA_DIR, "sample_submission.csv")

LABELS = [
    "is_toxic",
    "implicit_toxicity",
    "sarcasm",
    "contains_profanity",
    "threatening",
    "identity_attack",
    "sexual_content"
]

N_LABELS = len(LABELS)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 80)
print("DEVICE:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("=" * 80)


# ============================================================
# 2. SEED
# ============================================================

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

seed_everything(SEED)


# ============================================================
# 3. LOAD DATA
# ============================================================

print("\nLoading data...")

train_df = pd.read_csv(TRAIN_FILE)
valid_df = pd.read_csv(VALID_FILE)
test_df = pd.read_csv(TEST_FILE)
sample_df = pd.read_csv(SAMPLE_FILE)

print("Train:", train_df.shape)
print("Valid:", valid_df.shape)
print("Test :", test_df.shape)
print("Sample:", sample_df.shape)


# ============================================================
# 4. BASIC CHECK
# ============================================================

assert "text" in train_df.columns
assert "text" in valid_df.columns
assert "text" in test_df.columns
assert "ID" in test_df.columns

for col in LABELS:
    assert col in train_df.columns, f"Missing label: {col}"
    assert col in valid_df.columns, f"Missing label: {col}"

print("\nRequired columns verified.")


# ============================================================
# 5. COMBINE TRAIN + VALIDATION
# ============================================================

full_train = pd.concat([train_df, valid_df], axis=0, ignore_index=True)
print("\nCombined labeled data:", full_train.shape)


# ============================================================
# 6. TEXT CLEANING
# ============================================================

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = text.replace("\u200c", " ").replace("\u200d", " ")
    text = " ".join(text.split())
    return text

train_texts = full_train["text"].map(clean_text).tolist()
test_texts = test_df["text"].map(clean_text).tolist()

Y = full_train[LABELS].astype(int).values

print("\nLabel distribution:")
print(full_train[LABELS].sum())


# ============================================================
# 7. COMPETITION METRIC
# ============================================================

def competition_score(y_true, y_pred):
    macro_f1 = f1_score(y_true[:, 0], y_pred[:, 0], average="macro", zero_division=0)
    
    attr_scores = []
    for j in range(1, N_LABELS):
        score = f1_score(y_true[:, j], y_pred[:, j], average="binary", zero_division=0)
        attr_scores.append(score)

    subtask_b = np.mean(attr_scores)
    final_score = 0.5 * macro_f1 + 0.5 * subtask_b
    return final_score, macro_f1, subtask_b, attr_scores


# ============================================================
# 8. STRATIFIED FOLDS (Built-in Scikit-Learn)
# ============================================================

print("\nCreating folds...")
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
folds = list(skf.split(np.zeros(len(Y)), Y[:, 0]))
print("Number of folds:", len(folds))


# ============================================================
# 9. TF-IDF MODEL & OOF
# ============================================================

print("\n" + "=" * 80)
print("TF-IDF MODEL")
print("=" * 80)

tfidf_oof = np.zeros((len(train_texts), N_LABELS), dtype=np.float32)
tfidf_test_pred = np.zeros((len(test_texts), N_LABELS), dtype=np.float32)

for fold, (tr_idx, va_idx) in enumerate(folds):
    print(f"\nTF-IDF Fold {fold + 1}/{N_FOLDS}")
    
    tr_texts = [train_texts[i] for i in tr_idx]
    va_texts = [train_texts[i] for i in va_idx]
    
    word_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=60000, sublinear_tf=True)
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, max_features=60000, sublinear_tf=True)
    
    X_tr_w = word_vec.fit_transform(tr_texts)
    X_va_w = word_vec.transform(va_texts)
    X_te_w = word_vec.transform(test_texts)
    
    X_tr_c = char_vec.fit_transform(tr_texts)
    X_va_c = char_vec.transform(va_texts)
    X_te_c = char_vec.transform(test_texts)
    
    X_tr = hstack([X_tr_w, X_tr_c]).tocsr()
    X_va = hstack([X_va_w, X_va_c]).tocsr()
    X_te = hstack([X_te_w, X_te_c]).tocsr()
    
    y_tr = Y[tr_idx]

    for j, label in enumerate(LABELS):
        clf = LogisticRegression(C=TFIDF_C, class_weight="balanced", solver="liblinear", max_iter=2000)
        clf.fit(X_tr, y_tr[:, j])
        
        tfidf_oof[va_idx, j] = clf.predict_proba(X_va)[:, 1]
        tfidf_test_pred[:, j] += clf.predict_proba(X_te)[:, 1] / N_FOLDS


# ============================================================
# 10. TRANSFORMER DATASET & MODEL
# ============================================================

class ToxicDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=192):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        enc = self.tokenizer(
            text, truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt"
        )
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0)
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item


class TransformerClassifier(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.20)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(hidden_size // 2, N_LABELS)
        )

    def mean_pool(self, last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).float()
        summed = torch.sum(last_hidden_state * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.mean_pool(outputs.last_hidden_state, attention_mask)
        pooled = self.dropout(pooled)
        return self.classifier(pooled)


def make_pos_weights(y):
    weights = []
    for j in range(y.shape[1]):
        pos = np.sum(y[:, j] == 1)
        neg = np.sum(y[:, j] == 0)
        w = 1.0 if pos == 0 else neg / pos
        weights.append(np.clip(w, 1.0, 8.0))
    return torch.tensor(weights, dtype=torch.float, device=DEVICE)


# ============================================================
# 11. TRAIN TRANSFORMER FUNCTION
# ============================================================

def train_transformer_oof(model_name, model_tag):
    print("\n" + "=" * 80)
    print("MODEL:", model_tag)
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    oof_predictions = np.zeros((len(train_texts), N_LABELS), dtype=np.float32)
    test_predictions = np.zeros((len(test_texts), N_LABELS), dtype=np.float32)

    for fold, (tr_idx, va_idx) in enumerate(folds):
        print(f"\n{model_tag} | FOLD {fold + 1}/{N_FOLDS}")

        train_dataset = ToxicDataset([train_texts[i] for i in tr_idx], Y[tr_idx], tokenizer, MAX_LEN)
        valid_dataset = ToxicDataset([train_texts[i] for i in va_idx], Y[va_idx], tokenizer, MAX_LEN)
        test_dataset = ToxicDataset(test_texts, None, tokenizer, MAX_LEN)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
        valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

        model = TransformerClassifier(model_name).to(DEVICE)
        pos_weights = make_pos_weights(Y[tr_idx])
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        total_steps = len(train_loader) * BERT_EPOCHS
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

        scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

        best_score = -1.0
        best_state = None

        for epoch in range(BERT_EPOCHS):
            model.train()
            running_loss = 0.0

            for batch in train_loader:
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                    logits = model(input_ids, attention_mask)
                    loss = criterion(logits, labels)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                running_loss += loss.item()

            model.eval()
            val_probs = []
            with torch.no_grad():
                for batch in valid_loader:
                    input_ids = batch["input_ids"].to(DEVICE)
                    attention_mask = batch["attention_mask"].to(DEVICE)
                    logits = model(input_ids, attention_mask)
                    val_probs.append(torch.sigmoid(logits).cpu().numpy())

            val_probs = np.vstack(val_probs)
            val_score = competition_score(Y[va_idx], (val_probs >= 0.5).astype(int))[0]

            print(f"Epoch {epoch + 1}/{BERT_EPOCHS} | loss={running_loss/len(train_loader):.4f} | val_score={val_score:.5f}")

            if val_score > best_score:
                best_score = val_score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        model.load_state_dict(best_state)
        model.to(DEVICE)
        model.eval()

        # OOF Prediction
        val_probs = []
        with torch.no_grad():
            for batch in valid_loader:
                logits = model(batch["input_ids"].to(DEVICE), batch["attention_mask"].to(DEVICE))
                val_probs.append(torch.sigmoid(logits).cpu().numpy())
        oof_predictions[va_idx] = np.vstack(val_probs)

        # Test Prediction
        fold_test_probs = []
        with torch.no_grad():
            for batch in test_loader:
                logits = model(batch["input_ids"].to(DEVICE), batch["attention_mask"].to(DEVICE))
                fold_test_probs.append(torch.sigmoid(logits).cpu().numpy())
        test_predictions += np.vstack(fold_test_probs) / N_FOLDS

        del model, optimizer, scheduler, train_loader, valid_loader, test_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return oof_predictions, test_predictions


# ============================================================
# 12. RUN TRANSFORMER MODELS
# ============================================================

banglabert_oof, banglabert_test = train_transformer_oof("csebuetnlp/banglabert", "BanglaBERT")
banglishbert_oof, banglishbert_test = train_transformer_oof("csebuetnlp/banglishbert", "BanglishBERT")


# ============================================================
# 13. FAST & EFFICIENT WEIGHT AND THRESHOLD SEARCH
# ============================================================

print("\n" + "=" * 80)
print("SEARCHING BEST WEIGHTS & THRESHOLDS PER LABEL")
print("=" * 80)

P1, P2, P3 = banglabert_oof, banglishbert_oof, tfidf_oof
T1, T2, T3 = banglabert_test, banglishbert_test, tfidf_test_pred

best_w1 = np.zeros(N_LABELS, dtype=np.float32)
best_w2 = np.zeros(N_LABELS, dtype=np.float32)
best_w3 = np.zeros(N_LABELS, dtype=np.float32)
best_thresholds = np.zeros(N_LABELS, dtype=np.float32)

weights_list = []
for i in range(11):
    for j in range(11 - i):
        k = 10 - i - j
        weights_list.append((i / 10.0, j / 10.0, k / 10.0))

threshold_grid = np.arange(0.10, 0.91, 0.02)

for j, label in enumerate(LABELS):
    label_best_score = -1.0
    
    for w1, w2, w3 in weights_list:
        ensemble_prob = w1 * P1[:, j] + w2 * P2[:, j] + w3 * P3[:, j]
        
        for threshold in threshold_grid:
            pred = (ensemble_prob >= threshold).astype(int)
            
            if j == 0:
                score = f1_score(Y[:, j], pred, average="macro", zero_division=0)
            else:
                score = f1_score(Y[:, j], pred, average="binary", zero_division=0)
                
            if score > label_best_score:
                label_best_score = score
                best_w1[j], best_w2[j], best_w3[j] = w1, w2, w3
                best_thresholds[j] = threshold

    print(f"{label:25s} | W=({best_w1[j]:.1f}, {best_w2[j]:.1f}, {best_w3[j]:.1f}) | TH={best_thresholds[j]:.2f} | F1={label_best_score:.5f}")


# ============================================================
# 14. BUILD SUBMISSION
# ============================================================

ensemble_test = np.zeros_like(T1, dtype=np.float32)
test_pred = np.zeros_like(T1, dtype=int)

for j in range(N_LABELS):
    ensemble_test[:, j] = best_w1[j] * T1[:, j] + best_w2[j] * T2[:, j] + best_w3[j] * T3[:, j]
    test_pred[:, j] = (ensemble_test[:, j] >= best_thresholds[j]).astype(int)

submission = pd.DataFrame()
submission["ID"] = test_df["ID"].values

for j, label in enumerate(LABELS):
    submission[label] = test_pred[:, j]

# Output file for Kaggle
OUTPUT_FILE = "submission.csv"
submission.to_csv(OUTPUT_FILE, index=False)

print("\n" + "=" * 80)
print("SUBMISSION SAVED SUCCESSFULLY TO:", OUTPUT_FILE)
print("=" * 80)
print(submission.head(10))
