import pandas as pd
import re

# ========== Step 0: Path Configuration ==========
features_path = "path/to/ad_analysis_features.csv"
labels_path = "path/to/labels.csv"
output_path = "path/to/full_matched_feature_label_pairs.csv"

# ========== Step 1: Load Data ==========
X = pd.read_csv(features_path, encoding="ISO-8859-1")
y = pd.read_csv(labels_path, encoding="ISO-8859-1")

# Strip whitespace and normalize formats
X["ad_id"] = X["ad_id"].astype(str).str.strip()
y["Video Name"] = y["Video Name"].astype(str).str.strip()

# ========== Step 2: Exact Matching ==========
exact_matches = pd.merge(
    X,
    y.rename(columns={"Video Name": "ad_id"}),
    on="ad_id",
    how="inner"
)

# ========== Step 3: Extract Unmatched Feature Rows ==========
unmatched_X = X[~X["ad_id"].isin(exact_matches["ad_id"])].copy()

# ========== Step 4: Fuzzy Matching by First Keyword ==========
def extract_one_keyword(text):
    tokens = re.split(r'[_\-\s\.]', text.lower())
    tokens = [t for t in tokens if len(t) > 1 and t.isalnum()]
    return tokens[0] if len(tokens) >= 1 else None

matched_rows = []
matched_ad_ids = set()

for _, y_row in y.iterrows():
    video_id = y_row["Video Name"]
    if video_id in exact_matches["ad_id"].values:
        continue  # Skip already matched records

    keyword = extract_one_keyword(video_id)
    if not keyword:
        continue

    for _, x_row in unmatched_X.iterrows():
        ad_text = str(x_row["ad_id"]).lower()
        if keyword in ad_text and x_row["ad_id"] not in matched_ad_ids:
            row = x_row.to_dict()
            row["Final Sentiment"] = y_row["Final Sentiment"]
            matched_rows.append(row)
            matched_ad_ids.add(x_row["ad_id"])
            break  # Match only once per label row

# ========== Step 5: Merge Final Results ==========
fuzzy_matched_df = pd.DataFrame(matched_rows)
final_df = pd.concat([exact_matches, fuzzy_matched_df], ignore_index=True)

# ========== Step 6: Export as CSV ==========
final_df.to_csv(output_path, index=False, encoding="utf-8-sig")

# ========== Step 7: Log Summary ==========
print(f"✅ Matching complete: {len(exact_matches)} exact matches, {len(fuzzy_matched_df)} fuzzy matches.")
print(f"💾 Combined results saved to: {output_path}")
