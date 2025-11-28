import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

df_july    = pd.read_csv("top_july/topic_engagement.csv")
df_august  = pd.read_csv("top_august/topic_engagement.csv")
df_sept    = pd.read_csv("top_september/topic_engagement.csv")
df_oct     = pd.read_csv("top_october/topic_engagement.csv")

df_july["month"]   = "July"
df_august["month"] = "August"
df_sept["month"]   = "September"
df_oct["month"]    = "October"

df = pd.concat([df_july, df_august, df_sept, df_oct], ignore_index=True)
df = df[df["topic"] != -1]

month_order = ["July", "August", "September", "October"]
df["month"] = pd.Categorical(df["month"], categories=month_order, ordered=True)

df_counts = df[["topic", "label", "month", "post_count", "score_mean", "comments_mean"]]


pivot = df_counts.pivot(index="month", columns="label", values="post_count").fillna(0)
labels = pivot.columns.tolist()
plt.figure(figsize=(10,6))
for topic in labels:
    plt.plot(pivot.index, pivot[topic], marker="o", label=f"{topic}")

plt.xlabel("Month")
plt.ylabel("Post Count")
plt.title("Post Count per Topic Over Time")
plt.legend(title="Topic")
plt.grid(True)
plt.tight_layout()
plt.savefig("topic_trends_post_count.png")

pivot = df_counts.pivot(index="month", columns="label", values="score_mean").fillna(0)
labels = pivot.columns.tolist()
plt.figure(figsize=(10,6))
for topic in labels:
    plt.plot(pivot.index, pivot[topic], marker="o", label=f"{topic}")

plt.xlabel("Month")
plt.ylabel("Post Count")
plt.title("Post Count per Topic Over Time")
plt.legend(title="Topic")
plt.grid(True)
plt.tight_layout()
plt.savefig("topic_trends_score.png")

pivot = df_counts.pivot(index="month", columns="label", values="comments_mean").fillna(0)
labels = pivot.columns.tolist()
plt.figure(figsize=(10,6))
for topic in labels:
    plt.plot(pivot.index, pivot[topic], marker="o", label=f"{topic}")

plt.xlabel("Month")
plt.ylabel("Post Count")
plt.title("Post Count per Topic Over Time")
plt.legend(title="Topic")
plt.grid(True)
plt.tight_layout()
plt.savefig("topic_trends_comments.png")