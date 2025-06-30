import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# === Step 1: Load the dataset ===
file_path = "path/to/NOT_UPLOAD_MATCHED_DATA.csv"
df = pd.read_csv(file_path, encoding="ISO-8859-1")

# === Step 2: Keep only numeric features ===
df_numeric = df.select_dtypes(include=["int64", "float64"])

# === Step 3: Compute correlation matrix ===
correlation_matrix = df_numeric.corr()

# === Step 4: Save correlation matrix as CSV ===
output_csv_path = "path/to/feature_correlation_matrix.csv"
correlation_matrix.to_csv(output_csv_path, index=True)
print(f"Correlation matrix saved to: {output_csv_path}")

# === Step 5: Plot correlation heatmap ===
plt.figure(figsize=(14, 10))
sns.heatmap(
    correlation_matrix,
    cmap="coolwarm",         # Blue-orange contrast
    annot=False,             # Do not show numerical values
    fmt=".2f",
    square=True,
    linewidths=0.5,
    cbar_kws={"label": "Correlation Coefficient"}
)
plt.title("Feature Correlation Heatmap", fontsize=16)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
