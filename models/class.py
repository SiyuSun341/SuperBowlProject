import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# Optional model import
try:
    from catboost import CatBoostClassifier
    CatBoostInstalled = True
except ImportError:
    CatBoostInstalled = False
    print("⚠️ CatBoost not installed, will skip this model. Run pip install catboost to enable.")

# === Path Settings ===
features_path = "./video_feature_analysis_full_encoded.csv"
labels_path = "./video_level_sentiment_summary.csv"

# === Load and Clean Data ===
features_df = pd.read_csv(features_path, encoding='utf-8')
labels_df = pd.read_csv(labels_path, encoding='utf-8')
labels_df.columns = [c.strip() for c in labels_df.columns]

valid_mask = labels_df["Final Sentiment"].str.lower().str.strip() != "no comments"
labels_df = labels_df[valid_mask].reset_index(drop=True)
features_df = features_df[valid_mask].reset_index(drop=True)

# Label Encoding
label_column = "Final Sentiment"
le = LabelEncoder()
y = le.fit_transform(labels_df[label_column])
X = features_df.drop(columns=["video_id"], errors='ignore')

# Standardization + PCA Dimensionality Reduction
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=20)
X_pca = pca.fit_transform(X_scaled)

# Training + Test Split
X_train = X_pca
y_train = y
X_test, _, y_test, _ = train_test_split(X_train, y_train, test_size=0.7, random_state=42, stratify=y_train)

# === Model Configuration (Excluding Extra Trees / XGBoost) ===
model_configs = {
    "Random Forest": (RandomForestClassifier(random_state=42), {
        "n_estimators": [50],
        "max_depth": [5],
        "min_samples_leaf": [2]
    }),
    "Logistic Regression": (LogisticRegression(max_iter=1000), {
        "C": [0.1]
    }),
    "SVM": (SVC(), {
        "C": [1],
        "kernel": ["linear"]
    }),
    "Naive Bayes": (GaussianNB(), {}),
    "KNN": (KNeighborsClassifier(), {
        "n_neighbors": [5],
        "weights": ["uniform"]
    }),
    "MLP": (MLPClassifier(max_iter=500, early_stopping=True, random_state=42), {
        "hidden_layer_sizes": [(64,)],
        "activation": ["relu"],
        "alpha": [0.0001]
    })
}

if CatBoostInstalled:
    model_configs["CatBoost"] = (CatBoostClassifier(verbose=0, random_state=42), {
        "iterations": [100],
        "depth": [4],
        "learning_rate": [0.1],
        "l2_leaf_reg": [3]
    })

# === Model Training and Evaluation ===
results = []

for name, (model, param_grid) in model_configs.items():
    print(f"\n🔍 Training: {name}")
    grid = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    train_preds = best_model.predict(X_train)
    test_preds = best_model.predict(X_test)

    train_acc = accuracy_score(y_train, train_preds)
    test_acc = accuracy_score(y_test, test_preds)

    results.append({
        "Model": name,
        "Train Accuracy": round(train_acc, 4),
        "Test Accuracy": round(test_acc, 4)
    })

# === Output Accuracy Table ===
results_df = pd.DataFrame(results)
print("\n=== Final Accuracy Results ===")
print(results_df.to_string(index=False))

# === Plotting ===
plt.figure(figsize=(10, 6))
bar_width = 0.35
x = range(len(results_df))
plt.bar(x, results_df["Train Accuracy"], width=bar_width, label='Train Accuracy')
plt.bar([i + bar_width for i in x], results_df["Test Accuracy"], width=bar_width, label='Test Accuracy')
plt.xticks([i + bar_width / 2 for i in x], results_df["Model"], rotation=30)
plt.ylabel("Accuracy")
plt.title("Train vs Test Accuracy per Model")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()

# === Save Image ===
plot_path = "./accuracy_plot.png"
plt.savefig(plot_path)
plt.show()

print(f"\n✅ Accuracy comparison plot saved to: {plot_path}")