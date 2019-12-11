
from sklearn.model_selection import GridSearchCV
import gensim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import classification_report
import embeddingvectorizer
import json
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_validate

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier

from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier

from sklearn.model_selection import cross_val_score
from collections import defaultdict
from tabulate import tabulate
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

from nltk.corpus import stopwords
from string import punctuation
import nltk
import pandas as pd
from collections import Counter
import logging
from sklearn.linear_model import SGDClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


OUTPUTPATH = '../output/frames/'
PATH = '/Users/anne/surfdrive/uva/projects/RPA_KeepingScore/data/RPA_data_with_dictionaryscores.pkl'
PE = '/Users/anne/repos/embedding_models/RPA/w2v_300d2000-01-01_2018-12-31'

logging.info("\nLoading model...")
mod = gensim.models.Word2Vec.load(PE)
MDL = dict(zip(mod.wv.index2word, mod.wv.syn0))


df = pd.read_pickle(PATH)
# only keep newspaper data
df = df[df['type'] == 'newspaper']
# define frames
frames = ['attrresp', 'cnflct','ecnmc', 'hmnintrst']

# 2 = not present, set to zero (0 = not present, 1 = present)
df[frames] = df[frames].replace({2:0})
df['attrresp'].fillna(0, inplace=True)
#df.rename(columns= {'text_x': 'text'}, inplace=True)

df.rename(columns= {'text_clean': 'text'}, inplace=True)

train, test = train_test_split(df, random_state=42, test_size=0.3, shuffle=True)

x_train = train.text
x_test = test.text


logging.info('Start defining pipelines...\n\n')

SGD_tfidf_pipeline = Pipeline([
                 ('tfidf', TfidfVectorizer()),
                 ('clf', OneVsRestClassifier(SGDClassifier())),
                ])

SGD_count_pipeline = Pipeline([
                 ('count', CountVectorizer()),
                 ('clf', OneVsRestClassifier(SGDClassifier())),
                ])

SGD_count_embedding_pipeline = Pipeline([
                ("Embedding", embeddingvectorizer.EmbeddingCountVectorizer(MDL, 'mean')),
                ('clf', OneVsRestClassifier(SGDClassifier())),
                ])

SGD_tfidf_embedding_pipeline = Pipeline([
                ("Embedding", embeddingvectorizer.EmbeddingTfidfVectorizer(MDL, 'mean')),
                ('clf', OneVsRestClassifier(SGDClassifier())),
                ])

SVC_tfidf_pipeline = Pipeline([
                ('tfidf',  TfidfVectorizer()),
                ('clf', OneVsRestClassifier(SVC())),
            ])

SVC_count_pipeline = Pipeline([
                ('count',  CountVectorizer()),
                ('clf', OneVsRestClassifier(SVC())),
            ])

SVC_count_embedding_pipeline = Pipeline([
                ("Embedding", embeddingvectorizer.EmbeddingCountVectorizer(MDL, 'mean')),
                ('clf', OneVsRestClassifier(SVC())),
            ])

SVC_tfidf_embedding_pipeline = Pipeline([
                ("Embedding", embeddingvectorizer.EmbeddingTfidfVectorizer(MDL, 'mean')),
                ('clf', OneVsRestClassifier(SVC())),
            ])

PA_tfidf_pipeline = Pipeline([
                 ('tfidf', TfidfVectorizer()),
                 ('clf', OneVsRestClassifier(PassiveAggressiveClassifier())),
                ])

PA_count_pipeline = Pipeline([
                 ('count', CountVectorizer()),
                 ('clf', OneVsRestClassifier(PassiveAggressiveClassifier())),
                ])


PA_count_embedding_pipeline = Pipeline([
                ("Embedding", embeddingvectorizer.EmbeddingCountVectorizer(MDL, 'mean')),
                ('clf', OneVsRestClassifier(PassiveAggressiveClassifier())),
                ])

PA_tfidf_embedding_pipeline = Pipeline([
                ("Embedding", embeddingvectorizer.EmbeddingTfidfVectorizer(MDL, 'mean')),
                ('clf', OneVsRestClassifier(PassiveAggressiveClassifier())),
                ])


ET_tfidf_pipeline = Pipeline([
                 ('tfidf', TfidfVectorizer()),
                 ('clf', OneVsRestClassifier(ExtraTreesClassifier())),
                ])

ET_count_pipeline = Pipeline([
                 ('count', CountVectorizer()),
                 ('clf', OneVsRestClassifier(ExtraTreesClassifier())),
                ])

ET_count_embedding_pipeline = Pipeline([
                ("Embedding", embeddingvectorizer.EmbeddingCountVectorizer(MDL, 'mean')),
                ('clf', OneVsRestClassifier(ExtraTreesClassifier())),
                ])

ET_tfidf_embedding_pipeline = Pipeline([
                ("Embedding", embeddingvectorizer.EmbeddingTfidfVectorizer(MDL, 'mean')),
                ('clf', OneVsRestClassifier(ExtraTreesClassifier())),
                ])




all_models = [
("SGD tfidf", SGD_tfidf_pipeline ) ,

("SGD count",  SGD_count_pipeline  ) ,

("SGD count embedding",  SGD_count_embedding_pipeline ) ,

("SGD tfidf embedding", SGD_tfidf_embedding_pipeline ) ,

("SVC tfidf", SVC_tfidf_pipeline ) ,

("SVC count",  SVC_count_pipeline  ) ,

("SVC count embedding",  SVC_count_embedding_pipeline ) ,

("SVC tfidf embedding", SVC_tfidf_embedding_pipeline ) ,

("PA tfidf", PA_tfidf_pipeline ) ,

("PA count", PA_count_pipeline ) ,

("PA count embedding", PA_count_embedding_pipeline  ) ,

("PA tfidf embedding", PA_tfidf_embedding_pipeline ) ,

("ET tfidf", ET_tfidf_pipeline ) ,

("ET count", ET_count_pipeline ) ,

("ET count embedding", ET_count_embedding_pipeline ) ,

("ET tifdf embedding", ET_tfidf_embedding_pipeline )  ]


scoring = {'f1_micro' : 'f1_micro',
           'f1_macro' : 'f1_macro',
           'acc': 'accuracy',
           'prec_micro': 'precision_micro',
           'prec_macro': 'precision_macro',
           'rec_micro': 'recall_micro',
           'rec_macro': 'recall_macro'}


results = defaultdict(list)
for f in frames:
    logging.info("Starting cross validation for frame: {}\n\n\n".format(f))
    unsorted_scores = [(name, cross_validate(model, x_train, train[f], cv=5, scoring=scoring)) for name, model in all_models]
    results[f] += unsorted_scores

r = []
final_dict = {}
for k, v in results.items():
    print(k)
    for class_name, scoring in v:
        final_dict = {'frame': k,
                      'class_name': class_name,
                      'accuracy': scoring['test_acc'].mean(),
                      'precision_micro': scoring['test_prec_micro'].mean(),
                      'f1_micro': scoring['test_f1_micro'].mean() ,
                      'recall_macro': scoring['test_rec_macro'].mean() ,
                      'precision_macro': scoring['test_prec_macro'].mean(),
                      'f1_macro': scoring['test_f1_macro'].mean() ,
                      'recall_micro': scoring['test_rec_micro'].mean() }
        r.append(final_dict)

df = pd.DataFrame.from_dict(r)

df.sort_values(['frame', 'f1_micro'], ascending=False, inplace=True)

logging.info('Saving file....')

fname = '{}SML_results_text_cleaned'.format(OUTPUTPATH)
df.to_json(fname)


# Create the dictionary that defines the order for sorting
sorterIndex = dict(zip(order,range(len(order))))
# Generate a rank column that will be used to sort
# the dataframe numerically
df['Tm_Rank'] = df['classifier_updated'].map(sorterIndex)
df.sort_values(['Frame','Tm_Rank']).to_csv('../output/results_frames.csv')
