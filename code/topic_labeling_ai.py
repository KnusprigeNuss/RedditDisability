import json
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from bertopic import BERTopic
import pandas as pd
import sys

def make_prompt(top_words):
    prompt = f"""
You are an expert in labeling topics for academic research. Your labels should represent data on various informative tables.

Your task: Based on the keywords and example posts provided, create a clear topic label (2–5 words) that describes the central theme.
The context: All posts are from Reddit's r/disability subreddit.
The label should:
- Be concise (max 5 words)
- Use plain language
- Reflect the common subject across the keywords and examples

Keywords: {', '.join(top_words)}

Output ONLY the label. Do not include quotes, explanations, or punctuation.
        """.strip()
    return prompt



if len(sys.argv) > 1:
    arg_mode = sys.argv[1]

mode = "manual"
if arg_mode == "automatic":
    mode = "automatic"

if mode == "manual":
    print("Running topic labeling using mode \"manual\"")
else:
    print("Running topic labeling using mode \"automatic\"")

# model_sbert = SentenceTransformer('all-MiniLM-L6-v2')
# model_name = "google/flan-t5-large"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# bart_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
# flan = pipeline("text2text-generation", model=bart_model, tokenizer=tokenizer)
models = {
    "flan-t5": pipeline("text2text-generation", model="google/flan-t5-large"),
    # "flan-ul2": pipeline("text2text-generation", model="google/flan-ul2"),
    "alpaca": pipeline("text2text-generation", model="declare-lab/flan-alpaca-large"),
    # "mistral": pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.2")  
}

df = pd.read_csv("data/r_disability_with_topics.csv")
model = BERTopic.load("r_disability_bertopic_model")

# bart = pipeline("summarization", model="facebook/bart-large-cnn")
# mistral = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.2")

label_list = {}
for topic in model.get_topics():
    top_words = [word for word, _ in model.get_topic(topic)[:20]]  
    rep_docs = model.get_representative_docs(topic)[:2]  
    original_texts = df.loc[df['cleaned_text'].isin(rep_docs), 'text'].tolist()
    prompt = make_prompt(top_words)
    labels = {}
    for name, gen in models.items():
        result = gen(prompt, max_new_tokens=20)[0]['generated_text'].strip()
        labels[name] = result

    # print(labels)
    if mode == "manual":
        print(f"\n\n---------------Topic {topic}---------------")
        print(f"Topic {topic} LLM labels: {labels}")
        print(f"Topic {topic} top words: {top_words}")
        print("Topic {} example texts:\n********************Example Text 1********************\n{}".format(topic, "\n********************Example Text 2********************\n".join(original_texts)))
        user_label = input(f"Enter label for topic {topic} (suggested: {labels['alpaca']}): ").strip()
        label_list[topic] = user_label if user_label else labels['alpaca']
    else:
        label_list[topic] = labels['alpaca']


df['label'] = df['topic'].map(label_list)
df.to_csv("data/r_disability_with_topics_and_labels.csv", index=False)
