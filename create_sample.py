import pandas as pd
import os

# Load the complete processed dataset
df = pd.read_csv("data/kickstarter_us_filtered.csv")

# Select a representative sample of 500 campaigns
sample_df = (
    df.groupby("status", group_keys=False)
      .sample(frac=500 / len(df), random_state=42)
)

# Save the sample
os.makedirs("sample_data", exist_ok=True)

sample_df.to_csv(
    "sample_data/kickstarter_sample.csv",
    index=False
)

print(sample_df.shape)
print(sample_df["status"].value_counts())