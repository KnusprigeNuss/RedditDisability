import pandas as pd
from bertopic import BERTopic

df = pd.read_csv("data/r_disability_preproc.csv")
docs = df['cleaned_text'].tolist()
model = BERTopic()
topics, probs = model.fit_transform(docs)

df['topic'] = topics
df.to_csv("data/r_disability_with_topics.csv", index=False)

# printing some examples
print(model.get_topic_info())      
for topic_id in model.get_topic_freq()['Topic']:
    words = [word for word, _ in model.get_topic(topic_id)]
    print(f"Topic {topic_id} words: {words}")
print(model.get_topic(1))          
model.visualize_barchart(top_n_topics=15)

model.save("r_disability_bertopic_model")
