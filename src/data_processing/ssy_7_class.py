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

# Optional: Use CatBoost if available
try:
    from catboost import CatBoostClassifier
    CatBoostInstalled = True
except ImportError:
    CatBoostInstalled = False
    print("⚠️ CatBoost is not installed. To enable it, run: pip install catboost")

# ====== File Path Settings (replace with your own paths or use environment config) ======
features_path = "path/to/video_feature_analysis_full_encoded.csv"
sentiment_path = "path/to/reddit_ad_sentiment_summary.csv"
output_plot_path = "path/to/accuracy_plot.png"
output_sentiment_path = "path/to/reddit_ad_sentiment_summary_updated.csv"

# ====== Load Data ======
features_df = pd.read_csv(features_path)
sentiment_df = pd.read_csv(sentiment_path)

# ====== Recalculate Final Sentiment based on thresholds ======
def determine_sentiment(row):
    pos = row["Positive"]
    neg = row["Negative"]
    total = row["Total Comments"]
    threshold = total * 0.05
    if pos > neg and (pos - neg) > threshold:
        return "Positive"
    elif neg > pos and (neg - pos) > threshold:
        return "Negative"
    else:
        return "Neutral"

sentiment_df["Final Sentiment"] = sentiment_df.apply(determine_sentiment, axis=1)
sentiment_df.to_csv(output_sentiment_path, index=False, encoding='utf-8-sig')
print(f"✅ Sentiment labels updated and saved to: {output_sentiment_path}")

# ====== Keyword Extraction Function ======
def extract_key_words(text, drop_ext=False):
    text = str(text).lower().strip().replace("-", "_")
    if drop_ext and text.endswith(".mp4"):
        text = text[:-4]
    parts = text.split("_")
    if parts and parts[0].isdigit():
        parts = parts[1:]
    return parts[:5]  # First 5 key words only

features_df["key_words"] = features_df["video_id"].apply(lambda x: extract_key_words(x, drop_ext=True))
sentiment_df["key_words"] = sentiment_df["ad_name"].apply(lambda x: extract_key_words(x))

# ====== Fuzzy Matching between Feature and Sentiment Data ======
merged_rows = []
for _, f_row in features_df.iterrows():
    f_words = set(f_row["key_words"])
    for _, s_row in sentiment_df.iterrows():
        s_words = set(s_row["key_words"])
        if len(f_words.intersection(s_words)) >= 1:
            combined = pd.concat([f_row, s_row], axis=1).T
            merged_rows.append(combined)
            break

if not merged_rows:
    raise ValueError("❌ No matching ads found. Please relax match conditions or check naming format.")

merged_df = pd.concat(merged_rows).reset_index(drop=True)

# ====== Feature and Label Preparation ======
X = merged_df.drop(columns=["video_id", "ad_name", "Final Sentiment", "key_words"], errors='ignore')
y = LabelEncoder().fit_transform(merged_df["Final Sentiment"])
X = X.fillna(0)
y = y[:len(X)]

# ====== Standardization and Dimensionality Reduction ======
X_scaled = StandardScaler().fit_transform(X)
pca_n = min(20, X.shape[1])
X_pca = PCA(n_components=pca_n).fit_transform(X_scaled)

# ====== Train-Test Split ======
stratify_opt = y if len(set(y)) > 1 and len(y) >= 4 else None
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.3, random_state=42, stratify=stratify_opt
)

# ====== Define Classifiers and Hyperparameters ======
model_configs = {
    "Random Forest": (RandomForestClassifier(random_state=42), {
        "n_estimators": [20],
        "max_depth": [3],
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
        "n_neighbors": [6],
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
        "iterations": [20],
        "depth": [4],
        "learning_rate": [0.1],
        "l2_leaf_reg": [3]
    })

# ====== Train Models and Evaluate Accuracy ======
results = []
for name, (model, param_grid) in model_configs.items():
    print(f"\n🔍 Training: {name}")
    grid = GridSearchCV(model, param_grid, cv=2, scoring='accuracy', n_jobs=-1)
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

# ====== Output Accuracy Summary ======
results_df = pd.DataFrame(results)
print("\n=== Final Accuracy Results ===")
print(results_df.to_string(index=False))

# ====== Visualization ======
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
plt.savefig(output_plot_path)
plt.show()

print(f"\n✅ Accuracy plot saved to: {output_plot_path}")
