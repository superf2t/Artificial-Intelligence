# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.DataFrame(np.random.randn(1000, 4),
                     columns=['A', 'B', 'C', 'D'])
df = df.cumsum()
plt.figure();
df.plot();
plt.legend(loc='best')
plt.show()

train = pd.read_csv('IMDB/IMDB_data/labeledTrainData.tsv', sep='\t')
test = pd.read_csv('IMDB/IMDB_data/testData.tsv', sep='\t')
print(train.head())
print(test.head())

import matplotlib.pyplot as plt
import seaborn as sns
plt.figure()
sns.countplot(train['sentiment'])
plt.show()

from bs4 import BeautifulSoup
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
def review_to_text(review,remove_stopwords):
    raw_text = BeautifulSoup(review, 'html').get_text()
    letters = re.sub('[^a-zA-Z]', ' ', raw_text)
    words = letters.lower().split()
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        words = [w for w in words if w not in stop_words]
    return words

X_train =[]
for review in train['review']:
    X_train.append(' '.join(review_to_text(review,True)))
#必须空一行

X_test = []
for review in test['review']:
    X_test.append(' '.join(review_to_text(review, True)))
#必须空一行

y_train = train['sentiment']

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
pip_count = Pipeline([('count_vec', CountVectorizer(analyzer='word')),
                      ('mnb', MultinomialNB())])
pip_tfidf = Pipeline([('tfidf_vec', TfidfVectorizer(analyzer='word')),
                       ('mnb', MultinomialNB())])
params_count = {'count_vec__binary': [True, False], 'count_vec__ngram_range': [(1, 1), (1, 2)], 'mnb__alpha': [0.1, 1.0, 10.0]}
params_tfidf = {'tfidf_vec__binary': [True, False], 'tfidf_vec__ngram_range': [(1, 1), (1, 2)], 'mnb__alpha': [0.1, 1.0, 10.0]}

# gs_count = GridSearchCV(pip_count, params_count, cv=4, n_jobs=-1, verbose=1)
# gs_count.fit(X_train, y_train)
# print(gs_count.best_score_)
# print(gs_count.best_params_)
# count_y_predict = gs_count.predict(X_test)
#
# gs_tfidf = GridSearchCV(pip_tfidf, params_tfidf, cv=4, n_jobs=-1, verbose=1)
# gs_tfidf.fit(X_train, y_train)
# print(gs_tfidf.best_score_)
# print(gs_tfidf.best_params_)
# tfidf_y_predict = gs_tfidf.predict(X_test)
# submission_count = pd.DataFrame({'id': test['id'], 'sentiment': count_y_predict})
# submission_count.to_csv("IMDB_data/submission_count.csv", index=False)
#
# submission_tfidf = pd.DataFrame({"id": test['id'], 'sentiment': tfidf_y_predict})
# submission_tfidf.to_csv("IMDB_data/submission_tfidf.csv", index=False)

unlabled_train = pd.read_csv('IMDB/IMDB_data/unlabeledTrainData.tsv', sep='\t', quoting=3)

import nltk.data

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def review_to_sentences(review, tokenizer):
    raw_sentences = tokenizer.tokenize(review.strip())
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append(review_to_text(raw_sentence, False))
    return sentences

corpora = []
for review in unlabled_train['review']:
    corpora += review_to_sentences(review, tokenizer)
for review in train['review']:
    corpora += review_to_sentences(review, tokenizer)

num_features = 300
min_word_count = 20
num_workers = 4
context = 10
downsampling = 1e-3

from gensim.models.word2vec import Word2Vec
# model = Word2Vec(corpora, workers=num_workers,\
#                  size=num_features, min_count=min_word_count,\
#                  window=context, sample=downsampling)
# model.init_sims(replace=True)
model_name = 'IMDB_data/300feature.model'
# model.save(model_name)

model = Word2Vec.load(model_name)
print(model.most_similar('man'))

import numpy as np

def makeFeatureVec(words, model, num_features):
    feature_vec = np.zeros((num_features, ), dtype='float32')
    nwords = 0
    index2word_set = set(model.wv.index2word)
    for word in words:
        if word in index2word_set:
            nwords += 1
            feature_vec = np.add(feature_vec, model[word])
    if nwords > 0:
        feature_vec = np.divide(feature_vec, nwords)
    return feature_vec

def getAvgFeatureVecs(reviews, model, num_features):
    counter = 0
    review_feature_vecs = np.zeros((len(reviews), num_features), dtype='float32')
    for review in reviews:
        review_feature_vecs[counter] = makeFeatureVec(review, model, num_features)
        counter += 1
    return review_feature_vecs

clean_train_reviews = []
for review in train['review']:
    clean_train_reviews.append(review_to_text(review, remove_stopwords=True))

trainDataVecs = getAvgFeatureVecs(clean_train_reviews, model, num_features)

clean_test_reviews = []
for review in test['review']:
    clean_test_reviews.append(review_to_text(review,remove_stopwords=True))

testDataVecs = getAvgFeatureVecs(clean_test_reviews,model,num_features)

from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()
params_gbc =  {'n_estimatoy': [10, 100, 500],
               'learbing_rate': [0.01, 0.1, 1.0],
               'max_depth': [2., 3, 4]}
gs_gbc = GridSearchCV(estimator=gbc, param_grid=params_gbc, cv=4,n_jobs=-1)
gs_gbc.fit(trainDataVecs,y_train)
print(gs_gbc.best_score_)
print(gs_gbc.best_params_)

gbc_y_predict = gs_gbc.predict(X_test)
submission_gbc = pd.DataFrame({'id':test['id'],'sentiment':gbc_y_predict})
submission_gbc.to_csv('gbc.csv', index=False)