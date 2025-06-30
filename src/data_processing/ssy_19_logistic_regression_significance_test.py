import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder

# === Step 1: Load data ===
file_path = "path/to/NOT_UPLOAD_MATCHED_DATA.csv"
df = pd.read_csv(file_path, encoding="ISO-8859-1")

# === Step 2: Data preprocessing ===
# Remove rows with missing values
df = df.dropna()

# Extract target variable
y = df["Final Sentiment"]

# Drop non-numeric columns and the target
X = df.drop(columns=["Final Sentiment", "ï»¿Video Name"], errors='ignore')
X = X.select_dtypes(include=["int64", "float64", "bool"])

# Convert labels to binary: Positive → 1, others → 0
y = y.apply(lambda x: 1 if str(x).lower() == "positive" else 0)

# === Step 3: Add constant term (intercept) ===
X_const = sm.add_constant(X)

# === Step 4: Fit logistic regression model (using statsmodels) ===
model = sm.Logit(y, X_const)
result = model.fit(disp=0)  # Disable fitting output

# === Step 5: Extract coefficients and significance statistics ===
summary_df = pd.DataFrame({
    "feature": result.params.index,
    "coef": result.params.values,
    "p_value": result.pvalues,
    "conf_lower": result.conf_int()[0],
    "conf_upper": result.conf_int()[1]
})

# Optionally exclude the intercept term from output
summary_df = summary_df[summary_df["feature"] != "const"]

# === Step 6: Save results to CSV ===
summary_df.to_csv("logit_significance_result.csv", index=False)

# Optional: Print top 10 most significant features by p-value
print(summary_df.sort_values(by="p_value").head(10))
