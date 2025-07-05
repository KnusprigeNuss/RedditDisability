import pandas as pd
import re
from nltk.corpus import stopwords
import spacy


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  
    text = re.sub(r"\s+", " ", text)     
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  
    return text.strip() 

def remove_stopwords(text):
    return ' '.join([w for w in text.split() if w not in stop_words])

def lemmatize(text):
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc if not token.is_punct])


df = pd.read_csv("data/r_disability_praw.csv")
print(df.shape)
print(df.head())
print(df['text'].str.len().describe())

nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words('english'))

df['cleaned_text'] = df['text'].apply(clean_text)
df = df[df['cleaned_text'].str.len() >= 30]
df['cleaned_text'] = df['cleaned_text'].apply(remove_stopwords)
# df['cleaned_text'] = df['cleaned_text'].apply(lemmatize)

print(df.shape)
print(df.head())
print(df['text'].str.len().describe())


df.to_csv("data/r_disability_preproc.csv", index=False)
print(f"Saved {len(df)} posts.")