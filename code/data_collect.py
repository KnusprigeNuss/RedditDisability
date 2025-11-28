import praw
import pandas as pd
import sys

mode = "top"
if len(sys.argv) > 1:
    mode = sys.argv[1]

reddit = praw.Reddit(
    client_id='mD3zU4O2DhE38K_8e9KPog',
    client_secret='23x3q--19xnxkmJ9OOgPvHvN1uYlFA',
    user_agent='disability_analysis_script by /u/KnusprigeNuss'
)

posts = []
subreddit = reddit.subreddit('disability')

if mode == "new":
    print("Fetching posts from r/disability using mode \"new\"")
    submissions = subreddit.new(limit=2000)
elif mode == "hot":
    print("Fetching posts from r/disability using mode \"hot\"")
    submissions = subreddit.hot(limit=2000)
else:
    print("Fetching posts from r/disability using mode \"top\"")
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
# df['text'] = ' ' + df['selftext']

# df.to_csv("data/r_disability_praw.csv", index=False)
df.to_csv("test/data_top_october.csv", index=False)
print(f"Saved {len(df)} posts.")
