import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score

# === Step 1: Load data ===
y_raw = pd.read_csv("path/to/all_labels.csv", encoding="ISO-8859-1")
X_raw = pd.read_csv("path/to/ad_analysis_features_encoded.csv", encoding="ISO-8859-1")

# === Step 2: Clean and filter labels ===
# Remove invalid entries and rows without sentiment labels
y_raw = y_raw[y_raw["ï»¿Video Name"].apply(lambda x: isinstance(x, str) and x != "#NAME?")]
y_raw = y_raw.dropna(subset=["Final Sentiment"])

# === Step 3: Match features and labels using prefix ===
# Extract first 5 characters to use as matching keys
y_raw["prefix"] = y_raw["ï»¿Video Name"].astype(str).str[:5]
X_raw["prefix"] = X_raw["ad_id"].astype(str).str[:5]
common_prefixes = set(X_raw["prefix"]).intersection(set(y_raw["prefix"]))
X_matched = X_raw[X_raw["prefix"].isin(common_prefixes)].copy()
y_matched = y_raw[y_raw["prefix"].isin(common_prefixes)].copy()
merged = pd.merge(y_matched, X_matched, on="prefix", how="inner")

# === Step 4: Extract label and feature columns ===
y_final = merged["Final Sentiment"]
X_final = merged.drop(columns=["ï»¿Video Name", "ad_id", "prefix", "Final Sentiment"])

# === Step 5: Encode labels and scale features ===
le = LabelEncoder()
y_encoded = le.fit_transform(y_final)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_final)

# === Step 6: Split dataset ===
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.3, random_state=42
)

# === Step 7: Define classification models (with regularization to avoid overfitting) ===
models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=4,
        min_samples_split=15,
        min_samples_leaf=6,
        random_state=42
    ),
    "Logistic Regression": LogisticRegression(
        C=0.3,
        penalty="l2",
        max_iter=1000
    ),
    "SVM": SVC(
        C=0.3,
        kernel="rbf",
        gamma="scale"
    ),
    "Naive Bayes": GaussianNB(),
    "KNN": KNeighborsClassifier(
        n_neighbors=9,
        weights="distance",
        leaf_size=50
    ),
    "MLP": MLPClassifier(
        hidden_layer_sizes=(50,),
        alpha=0.05,
        learning_rate_init=0.01,
        early_stopping=True,
        max_iter=1000,
        random_state=42
    ),
    "CatBoost": CatBoostClassifier(
        iterations=100,
        depth=3,
        learning_rate=0.03,
        l2_leaf_reg=10,
        random_seed=42,
        verbose=0
    )
}

# === Step 8: Train and evaluate models ===
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    results.append((name, train_acc, test_acc))

# === Step 9: Display accuracy comparison ===
results_df = pd.DataFrame(results, columns=["Model", "Train Accuracy", "Test Accuracy"])
print("\n=== Model Accuracy Comparison (Regularized to Avoid Overfitting) ===")
print(results_df)
