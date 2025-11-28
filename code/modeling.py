import pandas as pd
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN

df = pd.read_csv("data/r_disability_preproc.csv")
df["cleaned_text"] = df["cleaned_text"].fillna("").astype(str)
docs = df["cleaned_text"].tolist()
# model = BERTopic(top_n_words=20, n_gram_range=(1, 2))
umap = UMAP(n_neighbors=17, n_components=3, metric='cosine', random_state=42)
hdb = HDBSCAN(min_cluster_size=23, min_samples=10, metric='euclidean', prediction_data=True)
model = BERTopic(top_n_words=20, min_topic_size=30, umap_model=umap, hdbscan_model=hdb, calculate_probabilities=True)
# model = BERTopic(top_n_words=20, min_topic_size=23, calculate_probabilities=True)
topics, probs = model.fit_transform(docs)
# topics = model.reduce_outliers(docs, topics, probabilities=probs, strategy="probabilities", threshold=0.1)
# topics = model.reduce_outliers(docs, topics, strategy="c-tf-idf", threshold=0.1)
topics = model.reduce_outliers(docs, topics, strategy="embeddings", threshold=0.5)
model.update_topics(docs, topics, top_n_words=20)
import numpy as np
print("Outliers after reduction:", np.sum(np.array(topics) == -1))

df['topic'] = topics
df.to_csv("data/r_disability_with_topics.csv", index=False)

topics_array = np.array(topics)

topic_freq = pd.Series(topics_array).value_counts().sort_index()
print("Cluster sizes after reduction:")
print(topic_freq)

print("Number of outliers (-1):", (topics_array == -1).sum())

# printing some examples
print("updated")
print(model.get_topic_info())      
for topic_id in model.get_topic_freq()['Topic']:
    words = [word for word, _ in model.get_topic(topic_id)]
    for word in words:
        print(f"Topic {topic_id} word: {word}")
print(model.get_topic(1))          
fig_bar = model.visualize_barchart(top_n_topics=20)
fig_bar.write_html("output/figures/topic_barchart.html")

model.save("r_disability_bertopic_model")

# Get topic frequencies (cluster sizes)
topic_freq = model.get_topic_freq()

print("Cluster sizes:")
print(topic_freq.head(20))   # first 20 topics with their sizes
