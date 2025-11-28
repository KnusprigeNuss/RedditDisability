import pandas as pd
import glob

files = glob.glob("fetched_posts/data*.csv")

dfs = [pd.read_csv(f) for f in files]
merged_df = pd.concat(dfs, ignore_index=True)

merged_df = merged_df.drop_duplicates()
merged_df = merged_df.drop_duplicates(subset=["id"])

merged_df.to_csv("merged_output.csv", index=False)

print(f"Total unique posts: {len(merged_df)}")
