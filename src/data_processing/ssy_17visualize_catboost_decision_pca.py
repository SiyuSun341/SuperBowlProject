import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ========== Path Configuration ==========
file_path = "path/to/NOT_UPLOAD_MATCHED_DATA.csv"

# ========== Step 1: Load and preprocess data ==========
df = pd.read_csv(file_path)
df = df.dropna()

# Assume the last column is the label
X = df.iloc[:, 1:-1]  # Skip first column (e.g., ad name)
y = df.iloc[:, -1]

# Encode labels to numeric values
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ========== Step 2: Train CatBoost model ==========
model = CatBoostClassifier(verbose=0)
model.fit(X_scaled, y_encoded)

# ========== Step 3: PCA + Decision Boundary Visualization ==========
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X_scaled)

# Make predictions
y_pred = model.predict(X_scaled)
y_proba = model.predict_proba(X_scaled)[:, 1]  # Probabilities for positive class

# Scatter plot of decision boundary in 2D PCA space
plt.figure(figsize=(10, 7))
plt.title("PCA Decision Boundary with CatBoost", fontsize=14)
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_proba, cmap='coolwarm', alpha=0.7)
plt.colorbar(scatter, label="Predicted Positive Probability")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid(True)
plt.tight_layout()
plt.savefig("pca_decision_boundary.png")
plt.show()

# ========== Step 4: Confusion Matrix and Classification Report ==========
conf_mat = confusion_matrix(y_encoded, y_pred)
report = classification_report(y_encoded, y_pred, target_names=label_encoder.classes_)
print("Confusion Matrix:\n", conf_mat)
print("\nClassification Report:\n", report)

# ========== Step 5: Confidence Distribution and Uncertainty Analysis ==========
low_conf_indices = (y_proba > 0.4) & (y_proba < 0.6)

plt.figure(figsize=(10, 6))
plt.hist(y_proba, bins=20, color='orange', edgecolor='black')
plt.axvspan(0.4, 0.6, color='gray', alpha=0.2, label='Uncertain Region (0.4–0.6)')
plt.title("Prediction Confidence Distribution")
plt.xlabel("Probability of Positive Class")
plt.ylabel("Number of Samples")
plt.legend()
plt.tight_layout()
plt.savefig("confidence_distribution.png")
plt.show()
