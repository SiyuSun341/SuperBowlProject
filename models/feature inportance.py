import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os

# === Path Settings ===
features_path = "./video_feature_analysis_full_encoded.csv"
labels_path = "./video_level_sentiment_summary.csv"
output_plot = "./feature_importance_top20.png"
output_excel = "./feature_importance_ranking.xlsx"

# === Load Data ===
features_df = pd.read_csv(features_path, encoding="utf-8")
labels_df = pd.read_csv(labels_path, encoding="utf-8")
labels_df.columns = labels_df.columns.str.strip()

# === Data Alignment Processing ===
# Remove rows with 'no comments' labels
valid_mask = labels_df["Final Sentiment"].str.lower().str.strip() != "no comments"
labels_df = labels_df[valid_mask].reset_index(drop=True)
features_df = features_df[valid_mask].reset_index(drop=True)

# === Label Encoding ===
le = LabelEncoder()
y = le.fit_transform(labels_df["Final Sentiment"])

# === Feature Preparation ===
X = features_df.drop(columns=["video_id"], errors="ignore")
feature_names = X.columns.tolist()

# === Data Split (70% train / 30% test) ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# === Model Training ===
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# === Extract Feature Importance ===
importances = rf.feature_importances_
feature_ranking = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False).reset_index(drop=True)

# === Save Table ===
feature_ranking.to_excel(output_excel, index=False)
print(f"✅ Feature importance ranking saved as: {output_excel}")

# === Visualize Top 20 Features ===
top_k = 20
plt.figure(figsize=(10, 7))
plt.barh(feature_ranking["Feature"][:top_k][::-1], feature_ranking["Importance"][:top_k][::-1])
plt.xlabel("Importance Score")
plt.title("Top 20 Feature Importances (Random Forest)")
plt.tight_layout()
plt.savefig(output_plot)
plt.show()

print(f"✅ Feature importance plot saved as: {output_plot}")