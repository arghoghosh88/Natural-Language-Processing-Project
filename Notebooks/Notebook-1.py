# ==================================================================
# BENGALI NLP SAFETY CHALLENGE 2026
# Kaggle-ready pipeline: word + char TF-IDF, per-label C search,
# per-label threshold tuning, scoring, and test submission.
#
# Works as-is in a Kaggle Notebook (reads from /kaggle/input/...,
# writes submission.csv to /kaggle/working/) and still falls back
# to the local/Colab working directory if you run it elsewhere.
# ==================================================================

# ------------------------------------------------------------------
# SECTION 1: INSTALL & IMPORT LIBRARIES
# (Kaggle notebooks already ship these -- this is a harmless no-op
# if they're already installed, and lets the same file run on
# Colab or a fresh local machine too.)
# ------------------------------------------------------------------
!pip install -q scikit-learn scipy pandas numpy

import os
import re
import glob
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
# SECTION 2: LOCATE & LOAD DATA (train.csv, validation.csv, test.csv)
#
# On Kaggle, competition/dataset files live under
#   /kaggle/input/<dataset-or-competition-name>/<file>.csv
# not in the notebook's own working directory. This helper checks,
# in order: the current directory, any /kaggle/input/*/<file>.csv
# match, and (if present) a Colab upload prompt. Output always
# goes to /kaggle/working/ when that directory exists, else "./".
# ------------------------------------------------------------------
IN_KAGGLE = os.path.isdir("/kaggle/input")
OUTPUT_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."

# Only treat this as a real Colab runtime if we're NOT on Kaggle --
# some environments (Kaggle included) ship an importable `google.colab`
# stub even though there's no actual file-picker/browser behind it,
# which previously caused files.upload() to hang forever.
IN_COLAB = False
if not IN_KAGGLE:
    try:
        from google.colab import files as colab_files
        IN_COLAB = True
    except ImportError:
        IN_COLAB = False


def find_csv(name):
    """Return the first matching path for `name`, searching the
    current directory first, then every /kaggle/input subfolder."""
    if os.path.exists(name):
        return name
    if IN_KAGGLE:
        matches = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
        if matches:
            return matches[0]
    return None


def load_csv(name):
    path = find_csv(name)
    if path is not None:
        print(f"Loaded {name} from: {path}")
        return pd.read_csv(path)
    if IN_COLAB:
        print(f"'{name}' not found. Please upload it now...")
        colab_files.upload()
        if os.path.exists(name):
            return pd.read_csv(name)
    if IN_KAGGLE:
        print("Contents of /kaggle/input:")
        for root, dirs, fnames in os.walk("/kaggle/input"):
            for f in fnames:
                print(" ", os.path.join(root, f))
    raise FileNotFoundError(
        f"Could not find '{name}'. On Kaggle: click 'Add Input' in the "
        f"notebook sidebar and attach the dataset/competition that "
        f"contains this file, then re-run this cell -- see the listing "
        f"above for what's currently attached. Locally: place the file "
        f"in the working directory."
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
# config) -- recomputed here so the comparison table reflects
# whatever data is actually loaded.
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
# word-boundary aware). Char n-grams help with Bengali/Banglish
# spelling variation and catch profanity/slur substrings word
# n-grams miss. Feature counts are capped to stay lightweight.
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
oof_probs = {}  # out-of-fold probabilities on train.csv, used for threshold tuning
for label in LABELS:
    y_train = train_df[label].values
    c, cv_score = best_C_for_label(X_train, y_train)
    chosen_C[label] = c

    # Out-of-fold predicted probabilities on TRAIN (3297 rows) -- this is
    # what we tune thresholds on below. validation.csv has only 500 rows,
    # and as few as 15 positives for the rarest labels, so a threshold
    # picked directly from it is mostly fitting noise and does not
    # generalize (this was the main cause of the validation-vs-leaderboard
    # gap). The larger, cross-validated OOF signal on train.csv is far
    # more stable.
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof_probs[label] = cross_val_predict(
        LogisticRegression(max_iter=1000, class_weight="balanced", C=c, solver="liblinear"),
        X_train, y_train, cv=skf, method="predict_proba",
    )[:, 1]

    model = LogisticRegression(max_iter=1000, class_weight="balanced", C=c, solver="liblinear")
    model.fit(X_train, y_train)
    models[label] = model
    print(f"  {label:20s}: best C = {c:<5} | CV Macro-F1 = {cv_score:.4f}")


# ------------------------------------------------------------------
# SECTION 8: PREDICT PROBABILITIES ON VALIDATION SET
# (used only to report/sanity-check the chosen thresholds below --
# NOT used to choose them)
# ------------------------------------------------------------------
val_probs = {label: models[label].predict_proba(X_val)[:, 1] for label in LABELS}


# ------------------------------------------------------------------
# SECTION 9: THRESHOLD TUNING
# Thresholds are chosen from the OOF probabilities on train.csv
# (Section 7), which is a much larger, less noisy signal than the
# 500-row validation set. validation.csv is then used only to
# report how each threshold performs, satisfying "use validation
# only for evaluation" while avoiding tuning on too few examples.
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
print("THRESHOLD TUNING (chosen from train.csv OOF predictions;")
print("reported below on validation.csv)")
print("=" * 60)

optimal_thresholds = {}
optimized_scores = {}
for label in LABELS:
    # Choose the threshold from the OOF (train) signal.
    t, _ = tune_threshold(train_df[label].values, oof_probs[label])
    optimal_thresholds[label] = t
    # Report how that threshold performs on validation.csv (not used to pick it).
    val_f1 = f1_score(val_df[label].values, (val_probs[label] >= t).astype(int), average="macro")
    optimized_scores[label] = val_f1
    print(f"  {label:20s}: threshold = {t:.2f} | validation Macro-F1 = {val_f1:.4f}")


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
# submission.csv is written to /kaggle/working/ on Kaggle so it
# shows up in the notebook's Output tab and can be submitted
# directly to the competition.
# ==================================================================
test_df = load_csv("test.csv")
test_df["clean_text"] = test_df["text"].apply(clean_text)

X_test_word = word_vectorizer.transform(test_df["clean_text"])
X_test_char = char_vectorizer.transform(test_df["clean_text"])
X_test = hstack([X_test_word, X_test_char]).tocsr()

pred_df = pd.DataFrame({"ID": test_df["ID"]})
for label in LABELS:
    probs = models[label].predict_proba(X_test)[:, 1]
    pred_df[label] = (probs >= optimal_thresholds[label]).astype(int)

# ------------------------------------------------------------------
# IMPORTANT: build the submission against sample_submission.csv's own
# ID list, not just test.csv's own row order. If the two files ever
# disagree on which IDs belong to the test set (as happened before --
# only 198/1000 IDs actually matched), silently using test.csv's IDs
# produces a file the grader doesn't recognize, which can crater the
# leaderboard score even when the model itself is fine.
# ------------------------------------------------------------------
try:
    sample_sub = load_csv("sample_submission.csv")
    missing_from_predictions = set(sample_sub["ID"]) - set(pred_df["ID"])
    extra_in_predictions = set(pred_df["ID"]) - set(sample_sub["ID"])

    if missing_from_predictions or extra_in_predictions:
        print("\n" + "!" * 60)
        print("WARNING: test.csv and sample_submission.csv ID sets differ.")
        print(f"  IDs in sample_submission.csv but NOT predicted: {len(missing_from_predictions)}")
        print(f"  IDs predicted but NOT in sample_submission.csv: {len(extra_in_predictions)}")
        print("  This most likely means test.csv / sample_submission.csv were")
        print("  downloaded from different versions of the competition data.")
        print("  Re-download BOTH files together from the Kaggle 'Data' tab.")
        print("!" * 60)

    # Reindex to sample_submission's exact ID list and order -- this is
    # what the grader will actually check row-for-row.
    submission = sample_sub[["ID"]].merge(pred_df, on="ID", how="left")
    n_unfilled = submission[LABELS[0]].isnull().sum()
    if n_unfilled > 0:
        print(f"\n{n_unfilled} required IDs had no matching test.csv row -- "
              f"filling with 0 (cannot predict without the row's text).")
        submission[LABELS] = submission[LABELS].fillna(0).astype(int)
except FileNotFoundError:
    print("\nsample_submission.csv not found -- falling back to test.csv's own ID order.")
    submission = pred_df

submission = submission[["ID"] + LABELS]

submission_path = os.path.join(OUTPUT_DIR, "submission.csv")
submission.to_csv(submission_path, index=False)

print("\n" + "=" * 60)
print(f"submission.csv written to: {submission_path}")
print(f"shape: {submission.shape}")
print("=" * 60)
print(submission.head())
