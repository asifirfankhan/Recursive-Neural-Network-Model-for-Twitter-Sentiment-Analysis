import pandas as pd
import sqlite3

conn = sqlite3.connect('Twitter Database.db')
c = conn.cursor()

sent_ver = pd.read_sql_query('SELECT * FROM SentimentData',conn)

c.close()
conn.close()

#%%   
sent_ver.head(10)

print('sent_ver.shape:', sent_ver.shape)

sent_ver.Sentiment.value_counts()

sent_ver.head()

sent = sent_ver[ ['Sentiment','Tweets'] ].copy()
sent.head(10)

X = sent.Tweets
y = sent.Sentiment 

X.head(10)
y.head(10)
#%%
#MultinomialNB Classifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
#%%
vect = TfidfVectorizer(stop_words='english', 
                       token_pattern=r'\b\w{2,}\b',
                       min_df=1, max_df=0.1,
                       ngram_range=(1,2))
mnb = MultinomialNB(alpha=2)

mnb_pipeline = make_pipeline(vect, mnb)

#%%
mnb_pipeline.named_steps
#%%
# Cross Validation
cv = cross_val_score(mnb_pipeline, X, y, scoring='accuracy', cv=10, n_jobs=-1)
print('\nMultinomialNB Classifier\'s Accuracy: %0.5f\n' % cv.mean())
