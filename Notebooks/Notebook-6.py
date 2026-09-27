# ==================================================================
# BENGALI NLP SAFETY CHALLENGE 2026
# Final pipeline: word + char TF-IDF, per-label C search,
# per-label threshold tuning, scoring, and test submission.
#
# Paste this whole file into a single Google Colab cell (or split
# at the "SECTION" comments into separate cells) and run top to
# bottom. train.csv / validation.csv / test.csv must be in the
# Colab working directory, or you'll be prompted to upload them.
# ==================================================================

# ------------------------------------------------------------------
# SECTION 1: INSTALL & IMPORT LIBRARIES
# ------------------------------------------------------------------
!pip install -q scikit-learn scipy pandas numpy

import os
import re
import warnings
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

LABELS = [
    "is_toxic",
    "implicit_toxicity",
    "sarcasm",
    "contains_profanity",
    "threatening",
    "identity_attack",
    "sexual_content",
]
SUBTASK_A_LABEL = "is_toxic"
SUBTASK_B_LABELS = [l for l in LABELS if l != SUBTASK_A_LABEL]


# ------------------------------------------------------------------
# SECTION 2: LOAD DATA (train.csv, validation.csv, test.csv)
# ------------------------------------------------------------------
try:
    from google.colab import files as colab_files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False


def load_csv(name):
    if os.path.exists(name):
        return pd.read_csv(name)
    if IN_COLAB:
        print(f"'{name}' not found in working directory. Please upload it now...")
        colab_files.upload()
        if os.path.exists(name):
            return pd.read_csv(name)
    raise FileNotFoundError(
        f"'{name}' not found. Place it in the Colab working directory or upload it."
    )


train_df = load_csv("train.csv")
val_df = load_csv("validation.csv")


# ------------------------------------------------------------------
# SECTION 3: BASIC DATASET CHECKS
# ------------------------------------------------------------------
print("=" * 60)
print("DATASET CHECKS")
print("=" * 60)
print("Train shape:", train_df.shape)
print("Validation shape:", val_df.shape)

print("\nMissing values (train):")
print(train_df.isnull().sum())
print("\nMissing values (validation):")
print(val_df.isnull().sum())

print("\nLabel distribution (train):")
for l in LABELS:
    pos = int(train_df[l].sum())
    print(f"  {l:20s}: {pos:5d} / {len(train_df)}  ({pos / len(train_df):.2%})")

assert train_df["text"].isnull().sum() == 0
assert val_df["text"].isnull().sum() == 0
for l in LABELS:
    assert set(train_df[l].unique()).issubset({0, 1}), f"{l} is not strictly binary"


# ------------------------------------------------------------------
# SECTION 4: BASELINE (your exact TF-IDF + Logistic Regression
# config) -- recomputed here on the real data so the final
# comparison table reflects this dataset, not an earlier estimate.
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("REPRODUCING YOUR ORIGINAL BASELINE")
print("=" * 60)

baseline_vectorizer = TfidfVectorizer(
    analyzer="word", ngram_range=(1, 2), min_df=2, max_features=30000, sublinear_tf=True
)
Xtr_base = baseline_vectorizer.fit_transform(train_df["text"])
Xva_base = baseline_vectorizer.transform(val_df["text"])

BASELINE_SCORES = {}
for l in LABELS:
    clf = LogisticRegression(max_iter=500, class_weight="balanced", solver="liblinear")
    clf.fit(Xtr_base, train_df[l].values)
    preds = clf.predict(Xva_base)
    BASELINE_SCORES[l] = f1_score(val_df[l].values, preds, average="macro")
    print(f"  {l:20s}: {BASELINE_SCORES[l]:.4f}")

BASELINE_SUBTASK_A = BASELINE_SCORES[SUBTASK_A_LABEL]
BASELINE_SUBTASK_B = float(np.mean([BASELINE_SCORES[l] for l in SUBTASK_B_LABELS]))
BASELINE_FINAL = (BASELINE_SUBTASK_A + BASELINE_SUBTASK_B) / 2
print(f"  Subtask A          : {BASELINE_SUBTASK_A:.4f}")
print(f"  Subtask B          : {BASELINE_SUBTASK_B:.4f}")
print(f"  Final Score        : {BASELINE_FINAL:.4f}")


# ------------------------------------------------------------------
# SECTION 5: TEXT CLEANING
# ------------------------------------------------------------------
def clean_text(text):
    """Light, safety-preserving cleanup: strip URLs/mentions, normalize
    whitespace. Deliberately does NOT strip punctuation or 'stopwords',
    since profanity/threat/identity-attack signals often live in short
    tokens and punctuation patterns that heavier cleanup would erase."""
    text = str(text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


train_df["clean_text"] = train_df["text"].apply(clean_text)
val_df["clean_text"] = val_df["text"].apply(clean_text)


# ------------------------------------------------------------------
# SECTION 6: FEATURE ENGINEERING
# Word-level TF-IDF (1-2 grams) + char-level TF-IDF (2-5 grams,
# word-boundary aware). Char n-grams are especially useful for
# Bengali/Banglish spelling variation and catch profanity/slur
# substrings word n-grams miss. Feature counts capped for Colab.
# ------------------------------------------------------------------
word_vectorizer = TfidfVectorizer(
    analyzer="word", ngram_range=(1, 2), min_df=2, max_features=30000, sublinear_tf=True
)
char_vectorizer = TfidfVectorizer(
    analyzer="char_wb", ngram_range=(2, 5), min_df=3, max_features=30000, sublinear_tf=True
)

# Fit ONLY on train.csv -> no leakage from validation/test.
X_train_word = word_vectorizer.fit_transform(train_df["clean_text"])
X_val_word = word_vectorizer.transform(val_df["clean_text"])
X_train_char = char_vectorizer.fit_transform(train_df["clean_text"])
X_val_char = char_vectorizer.transform(val_df["clean_text"])

X_train = hstack([X_train_word, X_train_char]).tocsr()
X_val = hstack([X_val_word, X_val_char]).tocsr()

print("\n" + "=" * 60)
print("FEATURE MATRIX SHAPES")
print("=" * 60)
print("X_train:", X_train.shape)
print("X_val  :", X_val.shape)


# ------------------------------------------------------------------
# SECTION 7: PER-LABEL MODEL TRAINING WITH REGULARIZATION SEARCH
# 5-fold stratified CV on TRAIN ONLY picks the best C per label;
# class_weight="balanced" handles the severe imbalance on labels
# like implicit_toxicity / threatening / sexual_content.
# ------------------------------------------------------------------
C_GRID = [0.1, 0.5, 1, 2, 5, 10]
N_FOLDS = 5


def best_C_for_label(X, y):
    best_score, best_c = -1.0, 1.0
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    for c in C_GRID:
        model = LogisticRegression(max_iter=1000, class_weight="balanced", C=c, solver="liblinear")
        cv_preds = cross_val_predict(model, X, y, cv=skf, method="predict")
        score = f1_score(y, cv_preds, average="macro")
        if score > best_score:
            best_score, best_c = score, c
    return best_c, best_score


print("\n" + "=" * 60)
print("PER-LABEL MODEL SELECTION (5-fold CV on train.csv only)")
print("=" * 60)

models = {}
chosen_C = {}
for label in LABELS:
    y_train = train_df[label].values
    c, cv_score = best_C_for_label(X_train, y_train)
    chosen_C[label] = c
    model = LogisticRegression(max_iter=1000, class_weight="balanced", C=c, solver="liblinear")
    model.fit(X_train, y_train)
    models[label] = model
    print(f"  {label:20s}: best C = {c:<5} | CV Macro-F1 = {cv_score:.4f}")


# ------------------------------------------------------------------
# SECTION 8: PREDICT PROBABILITIES ON VALIDATION SET
# ------------------------------------------------------------------
val_probs = {label: models[label].predict_proba(X_val)[:, 1] for label in LABELS}


# ------------------------------------------------------------------
# SECTION 9: THRESHOLD TUNING (validation set ONLY, 0.10 -> 0.90)
# ------------------------------------------------------------------
THRESHOLD_GRID = np.round(np.arange(0.10, 0.901, 0.02), 2)


def tune_threshold(y_true, probs):
    best_t, best_f1 = 0.50, -1.0
    for t in THRESHOLD_GRID:
        preds = (probs >= t).astype(int)
        score = f1_score(y_true, preds, average="macro")
        if score > best_f1:
            best_f1, best_t = score, t
    return best_t, best_f1


print("\n" + "=" * 60)
print("THRESHOLD TUNING (validation.csv only)")
print("=" * 60)

optimal_thresholds = {}
optimized_scores = {}
for label in LABELS:
    y_true = val_df[label].values
    t, f1 = tune_threshold(y_true, val_probs[label])
    optimal_thresholds[label] = t
    optimized_scores[label] = f1
    print(f"  {label:20s}: threshold = {t:.2f} | Macro-F1 = {f1:.4f}")


# ------------------------------------------------------------------
# SECTION 10: SUBTASK / FINAL SCORES (OPTIMIZED)
# ------------------------------------------------------------------
optimized_subtask_A = optimized_scores[SUBTASK_A_LABEL]
optimized_subtask_B = float(np.mean([optimized_scores[l] for l in SUBTASK_B_LABELS]))
optimized_final = (optimized_subtask_A + optimized_subtask_B) / 2


# ------------------------------------------------------------------
# SECTION 11: FINAL RESULT TABLE (BASELINE vs OPTIMIZED)
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("BASELINE vs OPTIMIZED")
print("=" * 60)
print(f"{'Label':<22}{'Baseline':<12}{'Optimized':<12}")
for label in LABELS:
    print(f"{label:<22}{BASELINE_SCORES[label]:<12.4f}{optimized_scores[label]:<12.4f}")
print("-" * 46)
print(f"{'Subtask A':<22}{BASELINE_SUBTASK_A:<12.4f}{optimized_subtask_A:<12.4f}")
print(f"{'Subtask B':<22}{BASELINE_SUBTASK_B:<12.4f}{optimized_subtask_B:<12.4f}")
print(f"{'Final Score':<22}{BASELINE_FINAL:<12.4f}{optimized_final:<12.4f}")
print("=" * 60)


# ==================================================================
# SECTION 12: FINAL SUBMISSION -- train final models -> predict
#             test.csv -> create submission.csv
#
# validation.csv was used ONLY for evaluation and threshold tuning
# above -- it was never used to fit the vectorizers or classifiers.
# The "final models" are exactly the models fit on train.csv in
# Section 7, applied with the thresholds tuned in Section 9.
# ==================================================================
test_df = load_csv("test.csv")
test_df["clean_text"] = test_df["text"].apply(clean_text)

X_test_word = word_vectorizer.transform(test_df["clean_text"])
X_test_char = char_vectorizer.transform(test_df["clean_text"])
X_test = hstack([X_test_word, X_test_char]).tocsr()

submission = pd.DataFrame()
submission["ID"] = test_df["ID"]
for label in LABELS:
    probs = models[label].predict_proba(X_test)[:, 1]
    submission[label] = (probs >= optimal_thresholds[label]).astype(int)

submission = submission[["ID"] + LABELS]
submission.to_csv("submission.csv", index=False)

print("\n" + "=" * 60)
print(f"submission.csv written with shape {submission.shape}")
print("=" * 60)
print(submission.head())
