import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score, roc_auc_score,
    log_loss, confusion_matrix, roc_curve, auc
)
import matplotlib.pyplot as plt

# === Step 1: 读取本地数据 ===
X_raw = pd.read_csv("D:/doctor/ad_analysis_features_encoded.csv", encoding="ISO-8859-1")
y_raw = pd.read_csv("D:/doctor/all_labels.csv", encoding="ISO-8859-1")

# === Step 2: 清洗标签数据 ===
y_raw = y_raw[y_raw["Video Name"].apply(lambda x: isinstance(x, str) and x != "#NAME?")]
y_raw = y_raw.dropna(subset=["Final Sentiment"])

# === Step 3: 构造前缀进行双向匹配 ===
y_raw["prefix"] = y_raw["Video Name"].astype(str).str[:5]
X_raw["prefix"] = X_raw["ad_id"].astype(str).str[:5]
common_prefixes = set(X_raw["prefix"]).intersection(set(y_raw["prefix"]))
X_matched = X_raw[X_raw["prefix"].isin(common_prefixes)].copy()
y_matched = y_raw[y_raw["prefix"].isin(common_prefixes)].copy()
merged = pd.merge(y_matched, X_matched, on="prefix", how="inner")

# === Step 4: 提取标签和特征 ===
y_final = merged["Final Sentiment"]
X_final = merged.drop(columns=["Video Name", "ad_id", "prefix", "Final Sentiment"])

# === Step 5: 标签编码 + 特征标准化 ===
le = LabelEncoder()
y_encoded = le.fit_transform(y_final)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_final)

# === Step 6: 数据集划分 ===
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.3, random_state=42
)

# === Step 7: 定义模型 ===
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_split=15, min_samples_leaf=6, random_state=42),
    "Logistic Regression": LogisticRegression(C=0.3, penalty="l2", max_iter=1000),
    "SVM": SVC(C=0.3, kernel="rbf", gamma="scale", probability=True),
    "Naive Bayes": GaussianNB(),
    "KNN": KNeighborsClassifier(n_neighbors=9, weights="distance", leaf_size=50),
    "MLP": MLPClassifier(hidden_layer_sizes=(50,), alpha=0.05, learning_rate_init=0.01, early_stopping=True, max_iter=1000, random_state=42),
    "CatBoost": CatBoostClassifier(iterations=100, depth=3, learning_rate=0.03, l2_leaf_reg=10, random_seed=42, verbose=0)
}

# === Step 8: 模型训练与评估 ===
results = []
plt.figure(figsize=(10, 8))
labels = np.unique(y_test)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)
    else:
        y_proba = None

    acc = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    # 多分类支持 AUC 与 logloss
    if y_proba is not None and y_proba.shape[1] == len(labels):
        auc_score = roc_auc_score(y_test, y_proba, multi_class='ovr', labels=labels)
        logloss = log_loss(y_test, y_proba, labels=labels)
    else:
        auc_score = np.nan
        logloss = np.nan

    # 特异度（仅限二分类）
    cm = confusion_matrix(y_test, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp)
    else:
        specificity = np.nan

    # ROC（仅限二分类）
    if y_proba is not None and len(labels) == 2:
        fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")

    results.append((name, acc, recall, f1, auc_score, logloss, specificity))

# === Step 9: 输出评估表 ===
results_df = pd.DataFrame(
    results,
    columns=["Model", "Accuracy", "Recall", "F1 Score", "AUC", "Log Loss", "Specificity"]
)
print("\n=== 模型评估指标对比 ===")
print(results_df)

# === Step 10: 绘制 ROC 曲线（仅二分类时） ===
plt.plot([0, 1], [0, 1], "k--")
plt.title("ROC Curve (Binary Classification Only)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()


results_df.to_csv("D:/doctor/model_metrics_results.csv", index=False)
