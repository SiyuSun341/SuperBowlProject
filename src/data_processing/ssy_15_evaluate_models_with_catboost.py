import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score, roc_auc_score,
    log_loss, confusion_matrix
)
from catboost import CatBoostClassifier

# === Step 1: Load data ===
data_path = "path/to/full_matched_feature_label_pairs.csv"
df = pd.read_csv(data_path, encoding="utf-8-sig")

# === Step 2: Remove rows labeled as "No Comment" ===
df = df[df[df.columns[-1]].str.strip().str.lower() != "no comment"]

# === Step 3: Separate features and labels ===
X = df.iloc[:, 1:-1]  # Use all columns except first (e.g., video name) and last (label)
y = df.iloc[:, -1]    # The last column contains the sentiment label

# === Step 4: Label encoding and feature scaling (except for CatBoost) ===
le = LabelEncoder()
y_encoded = le.fit_transform(y)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Step 5: Randomly split 30% of data as evaluation set ===
np.random.seed(42)
eval_indices = np.random.choice(len(X_scaled), size=int(0.3 * len(X_scaled)), replace=False)
train_indices = np.setdiff1d(np.arange(len(X_scaled)), eval_indices)

X_train, y_train = X_scaled[train_indices], y_encoded[train_indices]
X_eval, y_eval = X_scaled[eval_indices], y_encoded[eval_indices]

# === Step 6: Define models including CatBoost ===
models = {
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_split=10,
                                            min_samples_leaf=4, random_state=42),
    "Logistic Regression": LogisticRegression(C=1.0, penalty="l2", max_iter=1000),
    "SVM": SVC(C=2.0, kernel="rbf", gamma="scale", probability=True),
    "Naive Bayes": GaussianNB(),
    "KNN": KNeighborsClassifier(n_neighbors=7, weights="distance", leaf_size=30),
    "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), alpha=0.01, learning_rate_init=0.005,
                         early_stopping=True, max_iter=1000, random_state=42),
    "CatBoost": CatBoostClassifier(
        iterations=300,
        learning_rate=0.1,
        depth=6,
        verbose=0,
        random_seed=42
    )
}

# === Step 7: Train, predict, and evaluate each model ===
results = []
labels = np.unique(y_encoded)

for name, model in models.items():
    if name == "CatBoost":
        # Use raw features for CatBoost
        model.fit(X.iloc[train_indices], y_encoded[train_indices])
        y_pred_eval = model.predict(X.iloc[eval_indices])
        y_pred_train = model.predict(X.iloc[train_indices])
        y_proba_eval = model.predict_proba(X.iloc[eval_indices])
    else:
        model.fit(X_train, y_train)
        y_pred_eval = model.predict(X_eval)
        y_pred_train = model.predict(X_train)
        y_proba_eval = model.predict_proba(X_eval) if hasattr(model, "predict_proba") else None

    acc_train = accuracy_score(y_encoded[train_indices], y_pred_train)
    acc_eval = accuracy_score(y_encoded[eval_indices], y_pred_eval)
    recall_eval = recall_score(y_encoded[eval_indices], y_pred_eval, average='macro')
    f1_eval = f1_score(y_encoded[eval_indices], y_pred_eval, average='macro')

    if y_proba_eval is not None and y_proba_eval.shape[1] == len(labels):
        auc_eval = roc_auc_score(y_encoded[eval_indices], y_proba_eval, multi_class='ovr', labels=labels)
        logloss_eval = log_loss(y_encoded[eval_indices], y_proba_eval, labels=labels)
    else:
        auc_eval = np.nan
        logloss_eval = np.nan

    cm = confusion_matrix(y_encoded[eval_indices], y_pred_eval)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp)
    else:
        specificity = np.nan

    results.append((
        name, acc_train, acc_eval, recall_eval,
        f1_eval, auc_eval, logloss_eval, specificity
    ))

# === Step 8: Output performance table ===
results_df = pd.DataFrame(
    results,
    columns=[
        "Model", "Train Accuracy", "Eval Accuracy",
        "Eval Recall", "Eval F1", "Eval AUC",
        "Eval Log Loss", "Eval Specificity"
    ]
)

print("\n✅ Training & evaluation complete for all models:")
print(results_df)

# === Step 9: Save results to CSV ===
output_path = "path/to/model_metrics_with_catboost.csv"
results_df.to_csv(output_path, index=False)
print(f"\n📁 Evaluation results saved to: {output_path}")
