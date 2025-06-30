import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier

# ========== Step 1: Load data ==========
data_path = "path/to/NOT_UPLOAD_MATCHED_DATA.csv"
df = pd.read_csv(data_path, encoding="ISO-8859-1")

# ========== Step 2: Remove 'No Comment' labels ==========
df = df[df.iloc[:, -1] != "No Comment"]

# ========== Step 3: Extract features and labels ==========
X = df.iloc[:, 1:-1]  # Exclude video name (col 0) and label (last col)
y = df.iloc[:, -1]
feature_names = X.columns.tolist()

# ========== Step 4: Feature scaling ==========
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ========== Step 5: Train-test split for MLP ==========
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

# ========== Step 6: Feature importance plotting utility ==========
def plot_feature_importance(importances, model_name, feature_names, top_n=20):
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))
    plt.barh(importance_df["Feature"][::-1], importance_df["Importance"][::-1], color="orange")
    plt.title(f"{model_name} - Top {top_n} Feature Importances")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(f"{model_name.replace(' ', '_').lower()}_importance_top{top_n}.png", dpi=300)
    plt.close()

    importance_df.to_csv(f"{model_name.replace(' ', '_').lower()}_importance_top{top_n}.csv", index=False)
    print(f"{model_name} top {top_n} feature importances saved to CSV and PNG.")

# ========== Step 7: Train models and visualize importances ==========

# --- Random Forest ---
rf = RandomForestClassifier(random_state=42)
rf.fit(X_scaled, y)
plot_feature_importance(rf.feature_importances_, "Random Forest", feature_names)

# --- CatBoost ---
cat = CatBoostClassifier(verbose=0, random_seed=42)
cat.fit(X, y)  # CatBoost can handle raw features
plot_feature_importance(cat.get_feature_importance(), "CatBoost", feature_names)

# --- MLP with Permutation Importance ---
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
mlp.fit(X_train, y_train)
perm = permutation_importance(
    mlp, X_test, y_test, n_repeats=10, random_state=42, scoring='accuracy'
)
plot_feature_importance(perm.importances_mean, "MLP", feature_names)
