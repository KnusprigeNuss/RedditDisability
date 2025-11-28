import gradio as gr
import praw
import pandas as pd
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import spacy
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
import numpy as np
import os
from transformers import pipeline
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import textwrap


label_dict = {}
label_log = ""



def run_data_collect(mode):
    try:
        log = ""
        def log_and_yield(msg):
            nonlocal log
            log += msg + "\n"
            yield log

        reddit = praw.Reddit(
            client_id='mD3zU4O2DhE38K_8e9KPog',
            client_secret='23x3q--19xnxkmJ9OOgPvHvN1uYlFA',
            user_agent='disability_analysis_script by /u/KnusprigeNuss'
        )

        posts = []
        subreddit = reddit.subreddit('disability')

        if mode == "new":
            yield from log_and_yield("Fetching posts from r/disability using mode \"new\"")
            submissions = subreddit.new(limit=2000)
        elif mode == "hot":
            yield from log_and_yield("Fetching posts from r/disability using mode \"hot\"")
            submissions = subreddit.hot(limit=2000)
        else:
            yield from log_and_yield("Fetching posts from r/disability using mode \"top\"")
            submissions = subreddit.top(limit=2000)

        for submission in submissions: 
            if submission.selftext not in ['[removed]', '[deleted]']:
                posts.append({
                    'id': submission.id,
                    'title': submission.title,
                    'selftext': submission.selftext,
                    'score': submission.score,
                    'num_comments': submission.num_comments,
                    # 'created_utc': submission.created_utc
                })

        df = pd.DataFrame(posts)
        df['text'] = df['title'] + ' ' + df['selftext']

        df.to_csv("data/r_disability_praw.csv", index=False)
        yield from log_and_yield(f"Saved {len(df)} posts.")
        yield from log_and_yield("Data collection completed.")
    
    except Exception as e:
        yield f"Error during data collection: {str(e)}"



def run_preprocessing(clean_text, remove_spacy_stopwords, remove_sklearn_stopwords, lemmatize, min_text_length):
    try:
        nlp = spacy.load("en_core_web_sm")
        log = ""
        def log_and_yield(msg):
            nonlocal log
            log += msg + "\n"
            yield log

        def func_clean_text(text):
            text = text.lower()
            text = re.sub(r"http\S+", "", text)  
            text = re.sub(r"\s+", " ", text)     
            text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  
            return text.strip() 

        def func_remove_stopwords(text):
            return ' '.join([w for w in text.split() if w not in stop_words])

        def func_lemmatize(text):
            doc = nlp(text)
            return ' '.join([
                token.lemma_
                for token in doc
                if not token.is_punct and not (token.pos_ == "AUX")
            ])

        yield from log_and_yield("Running preprocessing")

        if not os.path.exists("data/r_disability_praw.csv"):
            yield "Error: Data not found. Please run Data Collection first."
            return

        df = pd.read_csv("data/r_disability_praw.csv")
        
        if min_text_length > 0:
            yield from log_and_yield(f"Filtering texts shorter than {min_text_length} characters")
            df = df[df['text'].str.len() >= min_text_length]

        if clean_text:
            yield from log_and_yield("Cleaning text")
            df['cleaned_text'] = df['text'].apply(func_clean_text)

        if lemmatize:
            yield from log_and_yield("Lemmatizing text")
            df['cleaned_text'] = df['cleaned_text'].apply(func_lemmatize)

        stop_words = set({"im", "disabled", "disability", "like", "people", "just", "know", "get", "think", "really", "one", "want", "time", "even", "make", "good", "way", "need", "see", "take", "also", "much", "still", "right", "something", "lot", "reddit", "subreddit", "post", "ive"})
        if remove_spacy_stopwords:
            yield from log_and_yield("Adding SPACY stopwords")
            stop_words = stop_words.union(set(nlp.Defaults.stop_words))
        if remove_sklearn_stopwords:
            yield from log_and_yield("Adding SKLEARN stopwords")
            stop_words = stop_words.union(set(ENGLISH_STOP_WORDS))
        if remove_spacy_stopwords or remove_sklearn_stopwords:
            yield from log_and_yield("Removing stopwords")
            df['cleaned_text'] = df['cleaned_text'].apply(func_remove_stopwords)

        df.to_csv("data/r_disability_preproc.csv", index=False)
        yield from log_and_yield(f"Saved {len(df)} posts")
        yield from log_and_yield("Preprocessing completed.")
    
    except Exception as e:
        yield f"Error during preprocessing: {str(e)}"



def run_modeling(hdb_min_cluster_size, hdb_min_samples, hdb_metric,
                umap_n_neighbors, umap_n_components, umap_metric,
                bert_min_topic_size, bert_top_n_words, prob_outlier_detection, prob_outlier_threshold,
                ctf_outlier_detection, ctf_outlier_threshold,
                emb_outlier_detection, emb_outlier_threshold):
    try:
        log = ""
        def log_and_yield(msg, fig=None):
            nonlocal log
            log += msg + "\n"
            if fig is None:
                yield log, None
            else:
                yield log, fig

        if not os.path.exists("data/r_disability_preproc.csv"):
            yield "Error: Data not found. Please run Preprocessing first.", None
            return
        
        df = pd.read_csv("data/r_disability_preproc.csv")
        df["cleaned_text"] = df["cleaned_text"].fillna("").astype(str)
        docs = df["cleaned_text"].tolist()

        yield from log_and_yield("Initializing UMAP with n_neighbors={}, n_components={}, metric={}".format(umap_n_neighbors, umap_n_components, umap_metric))
        umap = UMAP(n_neighbors=umap_n_neighbors, n_components=umap_n_components, metric=umap_metric, random_state=42)
        yield from log_and_yield("Initializing HDBSCAN with min_cluster_size={}, min_samples={}, metric={}".format(hdb_min_cluster_size, hdb_min_samples, hdb_metric))
        hdb = HDBSCAN(min_cluster_size=hdb_min_cluster_size, min_samples=hdb_min_samples, metric=hdb_metric, prediction_data=True)
        yield from log_and_yield("Initializing BERTopic with top_n_words={}, min_topic_size={}".format(bert_top_n_words, bert_min_topic_size))
        model = BERTopic(top_n_words=bert_top_n_words, min_topic_size=bert_min_topic_size, umap_model=umap, hdbscan_model=hdb, calculate_probabilities=True)
        topics, probs = model.fit_transform(docs)
        yield from log_and_yield("Fitting complete. Following topics were found:")
        yield from log_and_yield(str(model.get_topic_info()))
        if prob_outlier_detection:
            yield from log_and_yield("Reducing outliers using probabilities and threshold {}".format(prob_outlier_threshold))
            topics = model.reduce_outliers(docs, topics, probabilities=probs, strategy="probabilities", threshold=prob_outlier_threshold)
        curr_outliers = np.sum(np.array(topics) == -1)
        if ctf_outlier_detection and curr_outliers > 0:
            yield from log_and_yield("Reducing outliers using c-tf-idf and threshold {}".format(ctf_outlier_threshold))
            topics = model.reduce_outliers(docs, topics, strategy="c-tf-idf", threshold=ctf_outlier_threshold)
        curr_outliers = np.sum(np.array(topics) == -1)
        if emb_outlier_detection and curr_outliers > 0:
            yield from log_and_yield("Reducing outliers using embeddings and threshold {}".format(emb_outlier_threshold))
            topics = model.reduce_outliers(docs, topics, strategy="embeddings", threshold=emb_outlier_threshold)
        curr_outliers = np.sum(np.array(topics) == -1)
        if curr_outliers == 0:
            yield from log_and_yield("No outliers detected after reduction.")
        model.update_topics(docs, topics, top_n_words=20)

        df['topic'] = topics
        df.to_csv("data/r_disability_with_topics.csv", index=False)

        topics_array = np.array(topics)
        topic_freq = pd.Series(topics_array).value_counts().sort_index()
        yield from log_and_yield("Cluster sizes:")
        yield from log_and_yield(str(topic_freq))   
    
        fig_bar = model.visualize_barchart(top_n_topics=20)
        model.save("r_disability_bertopic_model")
        yield from log_and_yield("Modeling completed.", fig_bar)

    except Exception as e:
        yield f"Error during modeling: {str(e)}", None



def get_topic_info(topic_id):
    try:
        if not os.path.exists("r_disability_bertopic_model"):
            return "Error: Model not found. Please run Modeling first.", "", "", ""
        if not os.path.exists("data/r_disability_with_topics.csv"):
            return "Error: Data not found. Please run Modeling first.", "", "", ""

        model = BERTopic.load("r_disability_bertopic_model")
        df = pd.read_csv("data/r_disability_with_topics.csv")

        top_words = [w for w, _ in model.get_topic(topic_id)]
        keywords = ", ".join(top_words)
        rep_docs = model.get_representative_docs(topic_id)[:3] 
        original_texts = df.loc[df['cleaned_text'].isin(rep_docs), 'text'].tolist()

        return keywords, original_texts[0], original_texts[1], original_texts[2]
    
    except Exception as e:
        return f"Error during fetching topic info: {str(e)}", "", "", ""



def apply_label(topic_id, label):
    try:
        global label_dict
        def log_and_yield(msg):
            global label_log
            label_log += msg + "\n"
            yield label_log
        
        if not os.path.exists("data/r_disability_with_topics.csv"):
            yield "Error: Data not found. Please run Modeling first."
            return
        if not os.path.exists("r_disability_bertopic_model"):
            yield "Error: Model not found. Please run Modeling first."
            return

        df = pd.read_csv("data/r_disability_with_topics.csv")
        model = BERTopic.load("r_disability_bertopic_model")

        if not label:
            yield "Please enter a label."
            return
        label_dict[topic_id] = label
        label_dict[-1] = "Outlier"
        
        df['label'] = df['topic'].map(label_dict)
        df.to_csv("data/r_disability_with_topics_and_labels.csv", index=False)

        yield from log_and_yield(f"Saved label '{label}' for topic {topic_id}.")

    except Exception as e:
        yield f"Error during manual labeling: {str(e)}"



def make_prompt(top_words, example_texts):
    prompt = f"""
You are an expert in labeling topics for academic research. Your labels should represent data on various informative tables.

Your task: Based on the keywords and example posts provided, create a clear topic label (2–5 words) that describes the central theme.
The context: All posts are from Reddit's r/disability subreddit.
The label should:
- Be concise (max 5 words)
- Use plain language
- Reflect the common subject across the keywords and examples

Output ONLY the label. Do not include quotes, explanations, or punctuation.

Keywords: {', '.join(top_words)}
    """.strip()
    return prompt
# Example Posts: {" | ".join(example_texts)}



def run_auto_labeling():
    try:
        log = ""
        def log_and_yield(msg):
            nonlocal log
            log += msg + "\n"
            yield log
        model_alpaca = pipeline("text2text-generation", model="declare-lab/flan-alpaca-large")
        df = pd.read_csv("data/r_disability_with_topics.csv")
        model = BERTopic.load("r_disability_bertopic_model")

        label_list = {}
        for topic in model.get_topics():
            top_words = [word for word, _ in model.get_topic(topic)[:20]]
            rep_docs = model.get_representative_docs(topic)[:1]
            original_texts = df.loc[df['cleaned_text'].isin(rep_docs), 'text'].tolist()
            prompt = make_prompt(top_words, original_texts)
            result = model_alpaca(prompt, max_new_tokens=20)[0]['generated_text'].strip()
            label_list[topic] = result
            yield from log_and_yield(f"Topic {topic}: {result}")

        df['label'] = df['topic'].map(label_list)
        df.to_csv("data/r_disability_with_topics_and_labels.csv", index=False)
        yield from log_and_yield("Automatic labeling completed.")

    except Exception as e:
        yield f"Error during automatic labeling: {str(e)}"



def run_analysis(bubble_plot, score, comments, number, preview, download):
    try:
        plot1 = None
        plot2 = None
        plot3 = None
        plot4 = None        

        log = ""
        def log_and_yield(msg, fig1=None, fig2=None, fig3=None, fig4=None, p=None, down=None):
            nonlocal log
            log += msg + "\n"
            yield log, fig1, fig2, fig3, fig4, p, down

        if not os.path.exists("r_disability_bertopic_model"):
            yield "Error: Model not found. Please run Modeling first.", None, None, None, None
            return
        if not os.path.exists("data/r_disability_with_topics_and_labels.csv"):
            yield "Error: Data not found. Please run Labeling first.", None, None, None, None
            return

        model = BERTopic.load("r_disability_bertopic_model")
        df = pd.read_csv("data/r_disability_with_topics_and_labels.csv")

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

        cols = ['topic', 'label', 'score_mean', 'score_median', 'post_count', 'comments_mean', 'comments_median', 'keywords']
        engagement_stats = engagement_stats[cols]
        engagement_stats.to_csv("data/topic_engagement.csv", index=False)


        if number:
            df_no_outlier = df[df['topic'] != -1]
            df_label_counts = df_no_outlier['label'].value_counts().sort_values(ascending=False)
            wrapped_labels = ['\n'.join(textwrap.wrap(l, 25)) for l in df_label_counts.index]

            fig, ax = plt.subplots(figsize=(12, 8))
            ax.barh(wrapped_labels, df_label_counts.values)
            ax.set_title("Post Count per Topic")
            ax.set_xlabel("Number of Posts")
            ax.set_ylabel("Topic Label")
            fig.tight_layout()
            fig.savefig("figures/topic_frequency.png")
            plot4 = fig


        if score:
            df_avg_score = df_no_outlier.groupby('label')['score'].mean().sort_values(ascending=False)
            wrapped_labels = ['\n'.join(textwrap.wrap(l, 25)) for l in df_avg_score.index]

            fig, ax = plt.subplots(figsize=(12, 8))
            ax.barh(wrapped_labels, df_avg_score.values)
            ax.set_title("Average Score per Topic")
            ax.set_xlabel("Avg Score")
            ax.set_ylabel("Topic Label")
            fig.tight_layout()
            fig.savefig("figures/topic_avg_score.png")
            plot2 = fig


        if comments:
            df_avg_comments = df_no_outlier.groupby('label')['num_comments'].mean().sort_values(ascending=False)
            wrapped_labels = ['\n'.join(textwrap.wrap(l, 25)) for l in df_avg_comments.index]

            fig, ax = plt.subplots(figsize=(12, 8))
            ax.barh(wrapped_labels, df_avg_comments.values)
            ax.set_title("Average Comments per Topic")
            ax.set_xlabel("Average Comments")
            ax.set_ylabel("Topic Label")
            fig.tight_layout()
            fig.savefig("figures/topic_avg_comments.png")
            plot3 = fig


        if bubble_plot:
            plot_df = engagement_stats[engagement_stats['topic'] != -1].copy()

            fig, ax = plt.subplots(figsize=(12, 8))

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
                sizes=(100, 2000),
                ax=ax
            )

            max_len = 20  # show only first 20 chars
            truncated_labels = {label: (label[:max_len] + "…" if len(label) > max_len else label) 
                                for label in unique_labels}

            palette_dict = dict(zip(unique_labels, palette))

            handles = [
                mpatches.Patch(color=palette_dict[label], label=truncated_labels[label])
                for label in unique_labels
            ]

            ax.legend(
                handles=handles,
                title="Topic Label",
                bbox_to_anchor=(1.05, 1),
                loc="upper left"
            )
            ax.set_title("Topic Engagement: Score vs Comments")
            ax.set_xlabel("Average Score")
            ax.set_ylabel("Average Comments")
            fig.tight_layout()
            fig.savefig("figures/topic_engagement_bubble.png")
            plot1 = fig


        # ######## SINGLE PLOTS PER TOPIC ########
        # os.makedirs("output/figures/topic_scatterplots", exist_ok=True)

        # topic_labels = (
        #     df_no_outlier[['topic', 'label']]
        #     .drop_duplicates()
        #     .sort_values('topic')
        #     .set_index('topic')['label']
        #     .to_dict()
        # )

        # for topic_id, label in topic_labels.items():
        #     subset = df_no_outlier[df_no_outlier['topic'] == topic_id]

        #     plt.figure(figsize=(8,6))
        #     plt.scatter(
        #         subset['score'],
        #         subset['num_comments'],
        #         color="tab:blue",
        #         alpha=0.8
        #     )
        #     plt.xlabel("Score")
        #     plt.ylabel("Number of Comments")
        #     plt.title(f"Score vs Comments — {label} (Topic {topic_id}, n={len(subset)})")
        #     plt.tight_layout()
        #     plt.savefig(f"output/figures/topic_scatterplots/topic_{topic_id}.png")
        #     plt.close()

        yield from log_and_yield("Modeling completed.", plot1, plot2, plot3, plot4, engagement_stats, "data/topic_engagement.csv")

    except Exception as e:
        yield f"Error during analysis: {str(e)}"



with gr.Blocks() as demo:
    gr.Markdown("# Reddit Disability Project Control Panel")


    with gr.Tab("🔎 Data Collection"):
        gr.Markdown("### Data Collection Parameters")
        mode = gr.Radio(["top", "new", "hot"], label="Reddit Mode", value="top")
        collect_btn = gr.Button("Run Data Collection")
        status1 = gr.Textbox(label="Status", interactive=False)
        collect_btn.click(fn=run_data_collect, inputs=mode, outputs=status1)


    with gr.Tab("⚙️ Preprocessing"):
        gr.Markdown("### Preprocessing Parameters")
        min_text_length = gr.Slider(minimum=0, maximum=100, value=30, step=1, label="Minimum Text Length")
        clean_text = gr.Checkbox(label="Clean Text", value=True)
        lemmatize = gr.Checkbox(label="Lemmatize", value=True)
        remove_spacy_stopwords = gr.Checkbox(label="Remove SPACY Stopwords", value=True)
        remove_sklearn_stopwords = gr.Checkbox(label="Remove SKLEARN Stopwords", value=True)
        preprocess_btn = gr.Button("Run Preprocessing")
        status2 = gr.Textbox(label="Status", interactive=False)
        preprocess_btn.click(fn=run_preprocessing, inputs=[clean_text, remove_spacy_stopwords, remove_sklearn_stopwords, lemmatize, min_text_length], outputs=status2)


    with gr.Tab("📊 Topic Modeling"):
        gr.Markdown("### HDBSCAN Parameters")
        with gr.Row():
            hdb_min_cluster_size = gr.Slider(5, 50, value=23, step=1, label="min_cluster_size")
            hdb_min_samples = gr.Slider(1, 50, value=10, step=1, label="min_samples")
            hdb_metric = gr.Dropdown(["euclidean", "manhattan", "cosine"], value="euclidean", label="Metric")
        gr.Markdown("### UMAP Parameters")
        with gr.Row():
            umap_n_neighbors = gr.Slider(5, 50, value=17, step=1, label="n_neighbors")
            umap_n_components = gr.Slider(2, 20, value=3, step=1, label="n_components")
            umap_metric = gr.Dropdown(["euclidean", "manhattan", "cosine"], value="cosine", label="Metric")
        gr.Markdown("### BERTopic Parameters")
        with gr.Row():
            bert_min_topic_size = gr.Slider(5, 100, value=30, step=1, label="min_topic_size")
            bert_top_n_words = gr.Slider(5, 50, value=20, step=1, label="top_n_words")
        gr.Markdown("### Outlier handling")
        with gr.Row():
            prob_outlier_detection = gr.Checkbox(label="Probability Outlier Reduction", value=True)
            prob_outlier_threshold = gr.Slider(0, 1, value=0.1, step=0.01, label="Outlier Threshold")
        with gr.Row():
            ctf_outlier_detection = gr.Checkbox(label="C-TF-ID Outlier Reduction", value=True)
            ctf_outlier_threshold = gr.Slider(0, 1, value=0.1, step=0.01, label="Outlier Threshold")
        with gr.Row():
            emb_outlier_detection = gr.Checkbox(label="Embedding Outlier Reduction", value=True)
            emb_outlier_threshold = gr.Slider(0, 1, value=0.1, step=0.01, label="Outlier Threshold")

        model_btn = gr.Button("Run Modeling")
        bar_chart = gr.Plot(label="Topic Barchart")
        status3 = gr.Textbox(label="Status", interactive=False)

        model_btn.click(
            fn=run_modeling,
            inputs=[
                hdb_min_cluster_size, hdb_min_samples, hdb_metric,
                umap_n_neighbors, umap_n_components, umap_metric,
                bert_min_topic_size, bert_top_n_words,
                prob_outlier_detection, prob_outlier_threshold,
                ctf_outlier_detection, ctf_outlier_threshold,
                emb_outlier_detection, emb_outlier_threshold
            ],
            outputs=[status3, bar_chart]
        )


    with gr.Tab("🏷️ Labeling"):
        gr.Markdown("### Choose Labeling Mode")
        mode = gr.Radio(["Manual", "Automatic"], value="Manual", label="Mode")

        with gr.Column(visible=True) as manual_block:
            model = BERTopic.load("r_disability_bertopic_model")

            gr.Markdown("### Manual Labeling Information")
            topic_ids = sorted(model.get_topic_info()["Topic"].tolist())
            topic_ids = [tid for tid in topic_ids if tid != -1]
            topic_id = gr.Dropdown(choices=topic_ids, label="Topic ID", value=topic_ids[0] if topic_ids else None)
            keywords = gr.Textbox(label="Top Keywords", interactive=False)
            reps = gr.Textbox(label="Representative Post #1", lines=10, interactive=False)
            reps2 = gr.Textbox(label="Representative Post #2", lines=10, interactive=False)
            reps3 = gr.Textbox(label="Representative Post #3", lines=10, interactive=False)

            gr.Markdown("### Labels")
            with gr.Row():
                user_label = gr.Textbox(label="Insert your Label here:")
                btn_save = gr.Button("Save Label")
            status = gr.Textbox(label="Status", interactive=False)

            demo.load(fn=get_topic_info, inputs=topic_id, outputs=[keywords, reps, reps2, reps3])
            topic_id.change(fn=get_topic_info, inputs=topic_id, outputs=[keywords, reps, reps2, reps3])
            btn_save.click(fn=apply_label, inputs=[topic_id, user_label], outputs=status)

        with gr.Column(visible=False) as auto_block:
            gr.Markdown("### Automatic Labeling")
            auto_btn = gr.Button("Run Automatic Labeling")
            auto_status = gr.Textbox(label="Status", interactive=False)
            auto_btn.click(fn=run_auto_labeling, outputs=auto_status)

        def switch_mode(choice):
            return gr.update(visible=choice == "Manual"), gr.update(visible=choice == "Automatic")
        mode.change(fn=switch_mode, inputs=mode, outputs=[manual_block, auto_block])


    with gr.Tab("📈 Analysis"):
        gr.Markdown("### Analysis Selection")
        bubble = gr.Checkbox(label="Topic Engagement Bubble-Plot: Comments vs Score", value=True)
        score = gr.Checkbox(label="Average Score per Topic", value=True)
        comments = gr.Checkbox(label="Average Comments per Topic", value=True)
        number = gr.Checkbox(label="Number of Posts per Topic", value=True)
        scatter = gr.Checkbox(label="Topic Engagement Scatter-Plot per Topic", value=True)

        analysis_btn = gr.Button("Run Analysis")
        preview = gr.Dataframe(label="CSV Preview", interactive=False)
        download = gr.File(label="Download CSV")
        bubble_plot = gr.Plot(label="Topic Engagement Bubble-Plot: Comments vs Score")
        score_plot = gr.Plot(label="Average Score per Topic")
        comment_plot = gr.Plot(label="Average Comments per Topic")
        number_plot = gr.Plot(label="Number of Posts per Topic")
        
        status5 = gr.Textbox(label="Status", interactive=False)
        analysis_btn.click(fn=run_analysis, inputs=[bubble, score, comments, number, scatter], outputs=[status5, bubble_plot, score_plot, comment_plot, number_plot, preview, download])


demo.launch()