from bertopic import BERTopic
import pandas as pd
import textwrap
import seaborn as sns
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

df = pd.read_csv("data/r_disability_with_topics_and_labels.csv")
model = BERTopic.load("r_disability_bertopic_model")
os.makedirs("output/figures", exist_ok=True)

######## Engagement Stats ########
engagement_stats = df.groupby('topic').agg({
    'score': ['mean', 'median', 'count'],
    'num_comments': ['mean', 'median']
}).reset_index()

engagement_stats.columns = ['topic', 'score_mean', 'score_median', 'post_count', 'comments_mean', 'comments_median']
topic_labels = df[['topic', 'label']].drop_duplicates(subset=['topic'])
engagement_stats = engagement_stats.merge(topic_labels, on='topic', how='left')

topic_keywords = []
for topic_id in engagement_stats['topic']:
    words = model.get_topic(topic_id)
    keywords = ", ".join([w for w, _ in words])
    topic_keywords.append(keywords)
engagement_stats['keywords'] = topic_keywords

original_texts = []
for topic_id in engagement_stats['topic']:
    rep_docs = model.get_representative_docs(topic_id)[:2]
    original_texts2 = df.loc[df['cleaned_text'].isin(rep_docs), 'text'].tolist()
    original_texts.append(original_texts2)
engagement_stats['example_texts'] = original_texts

cols = ['topic', 'label', 'score_mean', 'score_median', 'post_count', 'comments_mean', 'comments_median', 'keywords', 'example_texts']
engagement_stats = engagement_stats[cols]
engagement_stats.to_csv("output/topic_engagement.csv", index=False)



######## Post Count per Topic ########
df_no_outlier = df[df['topic'] != -1]
df_label_counts = df_no_outlier['label'].value_counts().sort_values(ascending=False)
wrapped_labels = ['\n'.join(textwrap.wrap(l, 25)) for l in df_label_counts.index]

plt.figure(figsize=(12,8))
plt.barh(wrapped_labels, df_label_counts.values)
plt.title("Post Count per Topic")
plt.xlabel("Number of Posts")
plt.ylabel("Topic Label")
plt.tight_layout()
plt.savefig("output/figures/topic_frequency.png")
plt.clf()



##### Average Score per Topic ########
df_avg_score = df_no_outlier.groupby('label')['score'].mean().sort_values(ascending=False)
wrapped_labels = ['\n'.join(textwrap.wrap(l, 25)) for l in df_avg_score.index]

plt.figure(figsize=(12,8))
plt.barh(wrapped_labels, df_avg_score.values)
plt.title("Average Score per Topic")
plt.xlabel("Avg Score")
plt.ylabel("Topic Label")
plt.tight_layout()
plt.savefig("output/figures/topic_avg_score.png")
plt.clf()



######## Average Comments per Topic ########
df_avg_comments = df_no_outlier.groupby('label')['num_comments'].mean().sort_values(ascending=False)
wrapped_labels = ['\n'.join(textwrap.wrap(l, 25)) for l in df_avg_comments.index]

plt.figure(figsize=(12,8))
plt.barh(wrapped_labels, df_avg_comments.values)
plt.title("Average Comments per Topic")
plt.xlabel("Average Comments")
plt.ylabel("Topic Label")
plt.tight_layout()
plt.savefig("output/figures/topic_avg_comments.png")
plt.clf()



######## BUBBLE PLOT FOR TOPICS ########
plot_df = engagement_stats[engagement_stats['topic'] != -1].copy()
plt.figure(figsize=(12,8))

unique_labels = plot_df['label'].unique()
palette = sns.color_palette("tab20", len(unique_labels))
palette_dict = dict(zip(unique_labels, palette))

scatter = sns.scatterplot(
    data=plot_df,
    x="score_mean",
    y="comments_mean",
    hue="label",
    palette=palette_dict,
    alpha=0.9,
    legend=False,
    size="post_count",
    sizes=(100, 2000)
)

handles = [mpatches.Patch(color=palette_dict[label], label=label) for label in unique_labels]
plt.legend(
    handles=handles,
    title="Topic Label",
    bbox_to_anchor=(1.05, 1),
    loc="upper left"
)

plt.title("Topic Engagement: Score vs Comments")
plt.xlabel("Average Score")
plt.ylabel("Average Comments")
plt.tight_layout()
plt.savefig("output/figures/topic_engagement_bubble.png")
plt.clf()



######## SINGLE PLOTS PER TOPIC ########
os.makedirs("output/figures/topic_scatterplots", exist_ok=True)

topic_labels = (
    df_no_outlier[['topic', 'label']]
    .drop_duplicates()
    .sort_values('topic')
    .set_index('topic')['label']
    .to_dict()
)

for topic_id, label in topic_labels.items():
    subset = df_no_outlier[df_no_outlier['topic'] == topic_id]

    plt.figure(figsize=(8,6))
    plt.scatter(
        subset['score'],
        subset['num_comments'],
        color="tab:blue",
        alpha=0.8
    )
    plt.xlabel("Score")
    plt.ylabel("Number of Comments")
    plt.title(f"Score vs Comments — {label} (Topic {topic_id}, n={len(subset)})")
    plt.tight_layout()
    plt.savefig(f"output/figures/topic_scatterplots/topic_{topic_id}.png")
    plt.close()

