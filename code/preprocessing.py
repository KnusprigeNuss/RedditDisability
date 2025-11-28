import pandas as pd
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import spacy


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  
    text = re.sub(r"\s+", " ", text)     
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  
    # text = re.sub(r"[^a-zA-Z\s]", "", text)  
    return text.strip() 

def remove_stopwords(text):
    return ' '.join([w for w in text.split() if w not in stop_words])

def lemmatize(text):
    doc = nlp(text)
    return ' '.join([
        token.lemma_
        for token in doc
        if not token.is_punct and not (token.pos_ == "AUX")
    ])




# df = pd.read_csv("data/r_disability_praw.csv")
print("Running preprocessing...")
df = pd.read_csv("data/r_disability_praw.csv")
# print(df.head())
print(df['text'].str.len().describe())

nlp = spacy.load("en_core_web_sm")
# stop_words = set(stopwords.words('english'))
stop_words = nlp.Defaults.stop_words.union(set(ENGLISH_STOP_WORDS)).union({"im", "disabled", "disability", "like", "people", "just", "know", "get", "think", "really", "one", "want", "time", "even", "make", "good", "way", "need", "see", "take", "also", "much", "still", "right", "something", "lot", "reddit", "subreddit", "post"})
# stop_words = set(ENGLISH_STOP_WORDS).union(set(stopwords.words('english'))).union({"im", "disabled", "disability", "like", "people", "just", "know", "get", "think", "really", "one", "want", "time", "even", "make", "good", "way", "need", "see", "take", "also", "much", "still", "right", "something", "lot", "reddit", "subreddit", "post"})
# stop_words = set(ENGLISH_STOP_WORDS).union({"im"})

df['cleaned_text'] = df['text'].apply(clean_text)
df = df[df['text'].str.len() >= 30]
df['cleaned_text'] = df['cleaned_text'].apply(remove_stopwords)
# df['cleaned_text'] = df['cleaned_text'].apply(lemmatize)

print(df.shape)
print(df.head())
print(df['text'].str.len().describe())


df.to_csv("data/r_disability_preproc.csv", index=False)
print(f"Saved {len(df)} posts.")