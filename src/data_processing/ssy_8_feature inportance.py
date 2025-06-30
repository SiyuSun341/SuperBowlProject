import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from catboost import CatBoostClassifier
from sklearn.inspection import permutation_importance

# === Step 1: Load Feature and Label Data ===
X_raw = pd.read_csv("path/to/ad_analysis_features_encoded.csv", encoding="ISO-8859-1")
y_raw = pd.read_csv("path/to/all_labels.csv", encoding="ISO-8859-1")

# === Step 2: Clean Label Data ===
# Remove invalid or missing entries
y_raw = y_raw[y_raw["ï»¿Video Name"].apply(lambda x: isinstance(x, str) and x != "#NAME?")]
y_raw = y_raw.dropna(subset=["Final Sentiment"])

# === Step 3: Construct Prefix Keys for Matching ===
# Use the first 5 characters of ad_id and video name to align datasets
X_raw["prefix"] = X_raw["ad_id"].astype(str).str[:5]
y_raw["prefix"] = y_raw["ï»¿Video Name"].astype(str).str[:5]

# === Step 4: Merge Feature and Label Data ===
merged = pd.merge(X_raw, y_raw, on="prefix", how="inner")

# === Step 5: Drop Columns Not Used for Training ===
drop_cols = ["ad_id", "prefix", "ï»ƒVideo Name", "ï»¿Video Name", "Final Sentiment",
             "Positive", "Negative", "Neutral", "Total Comments"]
drop_cols = [col for col in drop_cols if col in merged.columns]

X_final = merged.drop(columns=drop_cols)
y_final = merged["Final Sentiment"]

# === Step 6: Label Encoding and Feature Scaling ===
le = LabelEncoder()
y_encoded = le.fit_transform(y_final)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_final)

# === Step 7: Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.3, random_state=42)

# === Step 8: Define Models ===
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=4,
    min_samples_split=15,
    min_samples_leaf=6,
    random_state=42
)

mlp = MLPClassifier(
    hidden_layer_sizes=(50,),
    alpha=0.05,
    learning_rate_init=0.01,
    early_stopping=True,
    max_iter=1000,
    random_state=42
)

cat = CatBoostClassifier(
    iterations=100,
    depth=3,
    learning_rate=0.03,
    l2_leaf_reg=10,
    random_seed=42,
    verbose=0
)

# === Step 9: Utility - Feature Importance Plotting (Top N) ===
def plot_feature_importance(importance, names, model_type, top_n=20):
    feature_importance = pd.DataFrame({'Feature': names, 'Importance': importance})
    feature_importance = feature_importance.sort_values(by="Importance", ascending=False).head(top_n).iloc[::-1]
    
    plt.figure(figsize=(12, 6))
    plt.barh(feature_importance['Feature'], feature_importance['Importance'], color='skyblue')
    plt.title(f"{model_type} - Top {top_n} Feature Importances")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.show()

# === Step 10: Train Models and Plot Feature Importances ===

# Random Forest
rf.fit(X_train, y_train)
plot_feature_importance(rf.feature_importances_, X_final.columns, "Random Forest")

# CatBoost
cat.fit(X_train, y_train)
plot_feature_importance(cat.feature_importances_, X_final.columns, "CatBoost")

# MLP with Permutation Importance
mlp.fit(X_train, y_train)
perm = permutation_importance(mlp, X_test, y_test, n_repeats=10, random_state=42, scoring='accuracy')
plot_feature_importance(perm.importances_mean, X_final.columns, "MLP (Permutation Importance)")
