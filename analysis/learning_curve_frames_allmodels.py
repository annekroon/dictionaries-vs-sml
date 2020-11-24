#!/usr/bin/env python3

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

from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Learning_rate_SML():

    def __init__(self, outputpath, datapath, path_to_embeddings):
        self.path_to_embeddings = path_to_embeddings
        self.outputpath =outputpath
        self.datapath = datapath
        self.frames = ['attrresp', 'cnflct','ecnmc', 'hmnintrst']
        self.frames_d = ['att_d' , 'cnflct_d', 'ecnm_d','hmninstr_d']
        self.df = self.Prep_df()
        self.train, self.test = train_test_split(self.df, random_state=42, test_size=0.2)
        model_300_15 = gensim.models.Word2Vec.load(self.path_to_embeddings + "w2v_size_300_window_15_negative_15")
        self.model_300_15 = dict(zip(model_300_15.wv.index2word, model_300_15.wv.syn0))
        model_300_10 = gensim.models.Word2Vec.load(self.path_to_embeddings + "w2v_size_300_window_10_negative_15")
        self.model_300_10 = dict(zip(model_300_10.wv.index2word, model_300_10.wv.syn0))

        model_100_15 = gensim.models.Word2Vec.load(self.path_to_embeddings + "w2v_size_100_window_15_negative_15")
        self.model_100_15 = dict(zip(model_100_15.wv.index2word, model_100_15.wv.syn0))
        model_100_10 = gensim.models.Word2Vec.load(self.path_to_embeddings + "w2v_size_100_window_10_negative_15")
        self.model_100_10 = dict(zip(model_100_10.wv.index2word, model_100_10.wv.syn0))

        self.all_models = self.Define_pipelines()
        self.training_sizes = list(range(10, 870, 10))
        #[100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 890]
        #[i for i in range(100, len(self.df), 100)]

    def Prep_df(self):

        df = pd.read_pickle(self.datapath)
        df = df[df['type'] == 'newspaper'] # only keep newspaper data
        df[self.frames] = df[self.frames].replace({2:0})# 2 = not present, set to zero (0 = not present, 1 = present)
        df['attrresp'].fillna(0, inplace=True)
        df.rename(columns= {'text_clean': 'text'}, inplace=True)

        return df

    def Load_embedding_model(self):

        for m in models:

            logging.info("\nLoading model...")

            mod = gensim.models.Word2Vec.load(self.path_to_embeddings)
            MDL = dict(zip(mod.wv.index2word, mod.wv.syn0))
            yield MDL

    def Define_pipelines(self):

        logging.info('Start defining pipelines...\n\n')

        SGD_tfidf_pipeline = Pipeline([
             ('tfidf', TfidfVectorizer()),
             ('clf', OneVsRestClassifier(SGDClassifier())),
            ])

        SGD_count_pipeline = Pipeline([
                         ('count', CountVectorizer()),
                         ('clf', OneVsRestClassifier(SGDClassifier())),
                        ])

        SGD_count_embedding_300_15_pipeline = Pipeline([
                        ("Embedding", embeddingvectorizer.EmbeddingCountVectorizer(self.model_300_15, 'mean')),
                        ('clf', OneVsRestClassifier(SGDClassifier())),
                        ])

        SGD_tfidf_embedding_300_15_pipeline = Pipeline([
                        ("Embedding", embeddingvectorizer.EmbeddingTfidfVectorizer(self.model_300_15, 'mean')),
                        ('clf', OneVsRestClassifier(SGDClassifier())),
                        ])

        SGD_count_embedding_300_15_pipeline_sum = Pipeline([
                        ("Embedding", embeddingvectorizer.EmbeddingCountVectorizer(self.model_300_15 , 'sum')),
                        ('clf', OneVsRestClassifier(SGDClassifier())),
                        ])

        SGD_tfidf_embedding_300_15_pipeline_sum = Pipeline([
                        ("Embedding", embeddingvectorizer.EmbeddingTfidfVectorizer(self.model_300_15 , 'sum')),
                        ('clf', OneVsRestClassifier(SGDClassifier())),
                        ])

        SGD_count_embedding_300_15_pipeline_max = Pipeline([
                        ("Embedding", embeddingvectorizer.EmbeddingCountVectorizer(self.model_300_15, 'max')),
                        ('clf', OneVsRestClassifier(SGDClassifier())),
                        ])

        SGD_tfidf_embedding_300_15_pipeline_max = Pipeline([
                        ("Embedding", embeddingvectorizer.EmbeddingTfidfVectorizer(self.model_300_15, 'max')),
                        ('clf', OneVsRestClassifier(SGDClassifier())),
                        ])


        SGD_count_embedding_300_10_pipeline = Pipeline([
                        ("Embedding", embeddingvectorizer.EmbeddingCountVectorizer(self.model_300_10, 'mean')),
                        ('clf', OneVsRestClassifier(SGDClassifier())),
                        ])

        SGD_tfidf_embedding_300_10_pipeline = Pipeline([
                        ("Embedding", embeddingvectorizer.EmbeddingTfidfVectorizer(self.model_300_10, 'mean')),
                        ('clf', OneVsRestClassifier(SGDClassifier())),
                        ])

        SGD_count_embedding_300_10_pipeline_sum = Pipeline([
                        ("Embedding", embeddingvectorizer.EmbeddingCountVectorizer(self.model_300_10 , 'sum')),
                        ('clf', OneVsRestClassifier(SGDClassifier())),
                        ])

        SGD_tfidf_embedding_300_10_pipeline_sum = Pipeline([
                        ("Embedding", embeddingvectorizer.EmbeddingTfidfVectorizer(self.model_300_10 , 'sum')),
                        ('clf', OneVsRestClassifier(SGDClassifier())),
                        ])

        SGD_count_embedding_300_10_pipeline_max = Pipeline([
                        ("Embedding", embeddingvectorizer.EmbeddingCountVectorizer(self.model_300_10, 'max')),
                        ('clf', OneVsRestClassifier(SGDClassifier())),
                        ])

        SGD_tfidf_embedding_300_10_pipeline_max = Pipeline([
                        ("Embedding", embeddingvectorizer.EmbeddingTfidfVectorizer(self.model_300_10, 'max')),
                        ('clf', OneVsRestClassifier(SGDClassifier())),
                        ])

        SGD_count_embedding_100_10_pipeline = Pipeline([
                        ("Embedding", embeddingvectorizer.EmbeddingCountVectorizer(self.model_100_10, 'mean')),
                        ('clf', OneVsRestClassifier(SGDClassifier())),
                        ])

        SGD_tfidf_embedding_100_10_pipeline = Pipeline([
                        ("Embedding", embeddingvectorizer.EmbeddingTfidfVectorizer(self.model_100_10, 'mean')),
                        ('clf', OneVsRestClassifier(SGDClassifier())),
                        ])

        SGD_count_embedding_100_10_pipeline_sum = Pipeline([
                        ("Embedding", embeddingvectorizer.EmbeddingCountVectorizer(self.model_100_10 , 'sum')),
                        ('clf', OneVsRestClassifier(SGDClassifier())),
                        ])

        SGD_tfidf_embedding_100_10_pipeline_sum = Pipeline([
                        ("Embedding", embeddingvectorizer.EmbeddingTfidfVectorizer(self.model_100_10 , 'sum')),
                        ('clf', OneVsRestClassifier(SGDClassifier())),
                        ])

        SGD_count_embedding_100_10_pipeline_max = Pipeline([
                        ("Embedding", embeddingvectorizer.EmbeddingCountVectorizer(self.model_100_10, 'max')),
                        ('clf', OneVsRestClassifier(SGDClassifier())),
                        ])

        SGD_tfidf_embedding_100_10_pipeline_max = Pipeline([
                        ("Embedding", embeddingvectorizer.EmbeddingTfidfVectorizer(self.model_100_10, 'max')),
                        ('clf', OneVsRestClassifier(SGDClassifier())),
                        ])


        SGD_count_embedding_100_15_pipeline = Pipeline([
                        ("Embedding", embeddingvectorizer.EmbeddingCountVectorizer(self.model_100_15, 'mean')),
                        ('clf', OneVsRestClassifier(SGDClassifier())),
                        ])

        SGD_tfidf_embedding_100_15_pipeline = Pipeline([
                        ("Embedding", embeddingvectorizer.EmbeddingTfidfVectorizer(self.model_100_15, 'mean')),
                        ('clf', OneVsRestClassifier(SGDClassifier())),
                        ])

        SGD_count_embedding_100_15_pipeline_sum = Pipeline([
                        ("Embedding", embeddingvectorizer.EmbeddingCountVectorizer(self.model_100_15 , 'sum')),
                        ('clf', OneVsRestClassifier(SGDClassifier())),
                        ])

        SGD_tfidf_embedding_100_15_pipeline_sum = Pipeline([
                        ("Embedding", embeddingvectorizer.EmbeddingTfidfVectorizer(self.model_100_15 , 'sum')),
                        ('clf', OneVsRestClassifier(SGDClassifier())),
                        ])

        SGD_count_embedding_100_15_pipeline_max = Pipeline([
                        ("Embedding", embeddingvectorizer.EmbeddingCountVectorizer(self.model_100_15, 'max')),
                        ('clf', OneVsRestClassifier(SGDClassifier())),
                        ])

        SGD_tfidf_embedding_100_15_pipeline_max = Pipeline([
                        ("Embedding", embeddingvectorizer.EmbeddingTfidfVectorizer(self.model_100_15, 'max')),
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

        SVC_count_embedding_300_15_pipeline = Pipeline([
                        ("Embedding", embeddingvectorizer.EmbeddingCountVectorizer(self.model_300_15, 'mean')),
                        ('clf', OneVsRestClassifier(SVC())),
                    ])

        SVC_tfidf_embedding_300_15_pipeline = Pipeline([
                        ("Embedding", embeddingvectorizer.EmbeddingTfidfVectorizer(self.model_300_15, 'mean')),
                        ('clf', OneVsRestClassifier(SVC())),
                    ])

        SVC_count_embedding_300_15_pipeline_sum = Pipeline([
                        ("Embedding", embeddingvectorizer.EmbeddingCountVectorizer(self.model_300_15, 'sum')),
                        ('clf', OneVsRestClassifier(SVC())),
                    ])

        SVC_tfidf_embedding_300_15_pipeline_sum = Pipeline([
                        ("Embedding", embeddingvectorizer.EmbeddingTfidfVectorizer(self.model_300_15, 'sum')),
                        ('clf', OneVsRestClassifier(SVC())),
                    ])

        SVC_count_embedding_300_15_pipeline_max = Pipeline([
                        ("Embedding", embeddingvectorizer.EmbeddingCountVectorizer(self.model_300_15, 'max')),
                        ('clf', OneVsRestClassifier(SVC())),
                    ])

        SVC_tfidf_embedding_300_15_pipeline_max = Pipeline([
                        ("Embedding", embeddingvectorizer.EmbeddingTfidfVectorizer(self.model_300_15, 'max')),
                        ('clf', OneVsRestClassifier(SVC())),
                    ])



        SVC_count_embedding_100_15_pipeline = Pipeline([
                        ("Embedding", embeddingvectorizer.EmbeddingCountVectorizer(self.model_100_15, 'mean')),
                        ('clf', OneVsRestClassifier(SVC())),
                    ])

        SVC_tfidf_embedding_100_15_pipeline = Pipeline([
                        ("Embedding", embeddingvectorizer.EmbeddingTfidfVectorizer(self.model_100_15, 'mean')),
                        ('clf', OneVsRestClassifier(SVC())),
                    ])

        SVC_count_embedding_100_15_pipeline_sum = Pipeline([
                        ("Embedding", embeddingvectorizer.EmbeddingCountVectorizer(self.model_100_15, 'sum')),
                        ('clf', OneVsRestClassifier(SVC())),
                    ])

        SVC_tfidf_embedding_100_15_pipeline_sum = Pipeline([
                        ("Embedding", embeddingvectorizer.EmbeddingTfidfVectorizer(self.model_100_15, 'sum')),
                        ('clf', OneVsRestClassifier(SVC())),
                    ])

        SVC_count_embedding_100_15_pipeline_max = Pipeline([
                        ("Embedding", embeddingvectorizer.EmbeddingCountVectorizer(self.model_100_15, 'max')),
                        ('clf', OneVsRestClassifier(SVC())),
                    ])

        SVC_tfidf_embedding_100_15_pipeline_max = Pipeline([
                        ("Embedding", embeddingvectorizer.EmbeddingTfidfVectorizer(self.model_100_15, 'max')),
                        ('clf', OneVsRestClassifier(SVC())),
                    ])



        SVC_count_embedding_100_10_pipeline = Pipeline([
                        ("Embedding", embeddingvectorizer.EmbeddingCountVectorizer(self.model_100_10, 'mean')),
                        ('clf', OneVsRestClassifier(SVC())),
                    ])

        SVC_tfidf_embedding_100_10_pipeline = Pipeline([
                        ("Embedding", embeddingvectorizer.EmbeddingTfidfVectorizer(self.model_100_10, 'mean')),
                        ('clf', OneVsRestClassifier(SVC())),
                    ])

        SVC_count_embedding_100_10_pipeline_sum = Pipeline([
                        ("Embedding", embeddingvectorizer.EmbeddingCountVectorizer(self.model_100_10, 'sum')),
                        ('clf', OneVsRestClassifier(SVC())),
                    ])

        SVC_tfidf_embedding_100_10_pipeline_sum = Pipeline([
                        ("Embedding", embeddingvectorizer.EmbeddingTfidfVectorizer(self.model_100_10, 'sum')),
                        ('clf', OneVsRestClassifier(SVC())),
                    ])

        SVC_count_embedding_100_10_pipeline_max = Pipeline([
                        ("Embedding", embeddingvectorizer.EmbeddingCountVectorizer(self.model_100_10, 'max')),
                        ('clf', OneVsRestClassifier(SVC())),
                    ])

        SVC_tfidf_embedding_100_10_pipeline_max = Pipeline([
                        ("Embedding", embeddingvectorizer.EmbeddingTfidfVectorizer(self.model_100_10, 'max')),
                        ('clf', OneVsRestClassifier(SVC())),
                    ])



        SVC_count_embedding_300_10_pipeline = Pipeline([
                        ("Embedding", embeddingvectorizer.EmbeddingCountVectorizer(self.model_300_10, 'mean')),
                        ('clf', OneVsRestClassifier(SVC())),
                    ])

        SVC_tfidf_embedding_300_10_pipeline = Pipeline([
                        ("Embedding", embeddingvectorizer.EmbeddingTfidfVectorizer(self.model_300_10, 'mean')),
                        ('clf', OneVsRestClassifier(SVC())),
                    ])

        SVC_count_embedding_300_10_pipeline_sum = Pipeline([
                        ("Embedding", embeddingvectorizer.EmbeddingCountVectorizer(self.model_300_10, 'sum')),
                        ('clf', OneVsRestClassifier(SVC())),
                    ])

        SVC_tfidf_embedding_300_10_pipeline_sum = Pipeline([
                        ("Embedding", embeddingvectorizer.EmbeddingTfidfVectorizer(self.model_300_10, 'sum')),
                        ('clf', OneVsRestClassifier(SVC())),
                    ])

        SVC_count_embedding_300_10_pipeline_max = Pipeline([
                        ("Embedding", embeddingvectorizer.EmbeddingCountVectorizer(self.model_300_10, 'max')),
                        ('clf', OneVsRestClassifier(SVC())),
                    ])

        SVC_tfidf_embedding_300_10_pipeline_max = Pipeline([
                        ("Embedding", embeddingvectorizer.EmbeddingTfidfVectorizer(self.model_300_10, 'max')),
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


        PA_count_embedding_300_10_pipeline = Pipeline([
                        ("Embedding", embeddingvectorizer.EmbeddingCountVectorizer(self.model_300_10, 'mean')),
                        ('clf', OneVsRestClassifier(PassiveAggressiveClassifier())),
                        ])

        PA_tfidf_embedding_300_10_pipeline = Pipeline([
                        ("Embedding", embeddingvectorizer.EmbeddingTfidfVectorizer(self.model_300_10, 'mean')),
                        ('clf', OneVsRestClassifier(PassiveAggressiveClassifier())),
                        ])

        PA_count_embedding_300_10_pipeline_sum = Pipeline([
                        ("Embedding", embeddingvectorizer.EmbeddingCountVectorizer(self.model_300_10, 'sum')),
                        ('clf', OneVsRestClassifier(PassiveAggressiveClassifier())),
                        ])

        PA_tfidf_embedding_300_10_pipeline_sum = Pipeline([
                        ("Embedding", embeddingvectorizer.EmbeddingTfidfVectorizer(self.model_300_10, 'sum')),
                        ('clf', OneVsRestClassifier(PassiveAggressiveClassifier())),
                        ])

        PA_count_embedding_300_10_pipeline_max = Pipeline([
                        ("Embedding", embeddingvectorizer.EmbeddingCountVectorizer(self.model_300_10, 'max')),
                        ('clf', OneVsRestClassifier(PassiveAggressiveClassifier())),
                        ])

        PA_tfidf_embedding_300_10_pipeline_max = Pipeline([
                        ("Embedding", embeddingvectorizer.EmbeddingTfidfVectorizer(self.model_300_10, 'max')),
                        ('clf', OneVsRestClassifier(PassiveAggressiveClassifier())),
                        ])



        PA_count_embedding_100_10_pipeline = Pipeline([
                        ("Embedding", embeddingvectorizer.EmbeddingCountVectorizer(self.model_100_10, 'mean')),
                        ('clf', OneVsRestClassifier(PassiveAggressiveClassifier())),
                        ])

        PA_tfidf_embedding_100_10_pipeline = Pipeline([
                        ("Embedding", embeddingvectorizer.EmbeddingTfidfVectorizer(self.model_100_10, 'mean')),
                        ('clf', OneVsRestClassifier(PassiveAggressiveClassifier())),
                        ])

        PA_count_embedding_100_10_pipeline_sum = Pipeline([
                        ("Embedding", embeddingvectorizer.EmbeddingCountVectorizer(self.model_100_10, 'sum')),
                        ('clf', OneVsRestClassifier(PassiveAggressiveClassifier())),
                        ])

        PA_tfidf_embedding_100_10_pipeline_sum = Pipeline([
                        ("Embedding", embeddingvectorizer.EmbeddingTfidfVectorizer(self.model_100_10, 'sum')),
                        ('clf', OneVsRestClassifier(PassiveAggressiveClassifier())),
                        ])

        PA_count_embedding_100_10_pipeline_max = Pipeline([
                        ("Embedding", embeddingvectorizer.EmbeddingCountVectorizer(self.model_100_10, 'max')),
                        ('clf', OneVsRestClassifier(PassiveAggressiveClassifier())),
                        ])

        PA_tfidf_embedding_100_10_pipeline_max = Pipeline([
                        ("Embedding", embeddingvectorizer.EmbeddingTfidfVectorizer(self.model_100_10, 'max')),
                        ('clf', OneVsRestClassifier(PassiveAggressiveClassifier())),
                        ])

#

        PA_count_embedding_100_15_pipeline = Pipeline([
                        ("Embedding", embeddingvectorizer.EmbeddingCountVectorizer(self.model_100_15, 'mean')),
                        ('clf', OneVsRestClassifier(PassiveAggressiveClassifier())),
                        ])

        PA_tfidf_embedding_100_15_pipeline = Pipeline([
                        ("Embedding", embeddingvectorizer.EmbeddingTfidfVectorizer(self.model_100_15, 'mean')),
                        ('clf', OneVsRestClassifier(PassiveAggressiveClassifier())),
                        ])

        PA_count_embedding_100_15_pipeline_sum = Pipeline([
                        ("Embedding", embeddingvectorizer.EmbeddingCountVectorizer(self.model_100_15, 'sum')),
                        ('clf', OneVsRestClassifier(PassiveAggressiveClassifier())),
                        ])

        PA_tfidf_embedding_100_15_pipeline_sum = Pipeline([
                        ("Embedding", embeddingvectorizer.EmbeddingTfidfVectorizer(self.model_100_15, 'sum')),
                        ('clf', OneVsRestClassifier(PassiveAggressiveClassifier())),
                        ])

        PA_count_embedding_100_15_pipeline_max = Pipeline([
                        ("Embedding", embeddingvectorizer.EmbeddingCountVectorizer(self.model_100_15, 'max')),
                        ('clf', OneVsRestClassifier(PassiveAggressiveClassifier())),
                        ])

        PA_tfidf_embedding_100_15_pipeline_max = Pipeline([
                        ("Embedding", embeddingvectorizer.EmbeddingTfidfVectorizer(self.model_100_15, 'max')),
                        ('clf', OneVsRestClassifier(PassiveAggressiveClassifier())),
                        ])

#

        PA_count_embedding_300_15_pipeline = Pipeline([
                        ("Embedding", embeddingvectorizer.EmbeddingCountVectorizer(self.model_300_15, 'mean')),
                        ('clf', OneVsRestClassifier(PassiveAggressiveClassifier())),
                        ])

        PA_tfidf_embedding_300_15_pipeline = Pipeline([
                        ("Embedding", embeddingvectorizer.EmbeddingTfidfVectorizer(self.model_300_15, 'mean')),
                        ('clf', OneVsRestClassifier(PassiveAggressiveClassifier())),
                        ])

        PA_count_embedding_300_15_pipeline_sum = Pipeline([
                        ("Embedding", embeddingvectorizer.EmbeddingCountVectorizer(self.model_300_15, 'sum')),
                        ('clf', OneVsRestClassifier(PassiveAggressiveClassifier())),
                        ])

        PA_tfidf_embedding_300_15_pipeline_sum = Pipeline([
                        ("Embedding", embeddingvectorizer.EmbeddingTfidfVectorizer(self.model_300_15, 'sum')),
                        ('clf', OneVsRestClassifier(PassiveAggressiveClassifier())),
                        ])

        PA_count_embedding_300_15_pipeline_max = Pipeline([
                        ("Embedding", embeddingvectorizer.EmbeddingCountVectorizer(self.model_300_15, 'max')),
                        ('clf', OneVsRestClassifier(PassiveAggressiveClassifier())),
                        ])

        PA_tfidf_embedding_300_15_pipeline_max = Pipeline([
                        ("Embedding", embeddingvectorizer.EmbeddingTfidfVectorizer(self.model_300_15, 'max')),
                        ('clf', OneVsRestClassifier(PassiveAggressiveClassifier())),
                        ])


#

        ET_tfidf_pipeline = Pipeline([
                         ('tfidf', TfidfVectorizer()),
                         ('clf', OneVsRestClassifier(ExtraTreesClassifier())),
                        ])

        ET_count_pipeline = Pipeline([
                         ('count', CountVectorizer()),
                         ('clf', OneVsRestClassifier(ExtraTreesClassifier())),
                        ])

        ET_count_embedding_300_15_pipeline = Pipeline([
                        ("Embedding", embeddingvectorizer.EmbeddingCountVectorizer(self.model_300_15, 'mean')),
                        ('clf', OneVsRestClassifier(ExtraTreesClassifier())),
                        ])

        ET_tfidf_embedding_300_15_pipeline = Pipeline([
                        ("Embedding", embeddingvectorizer.EmbeddingTfidfVectorizer(self.model_300_15, 'mean')),
                        ('clf', OneVsRestClassifier(ExtraTreesClassifier())),
                        ])


        ET_count_embedding_300_15_pipeline_sum = Pipeline([
                        ("Embedding", embeddingvectorizer.EmbeddingCountVectorizer(self.model_300_15, 'sum')),
                        ('clf', OneVsRestClassifier(ExtraTreesClassifier())),
                        ])

        ET_tfidf_embedding_300_15_pipeline_sum = Pipeline([
                        ("Embedding", embeddingvectorizer.EmbeddingTfidfVectorizer(self.model_300_15, 'sum')),
                        ('clf', OneVsRestClassifier(ExtraTreesClassifier())),
                        ])

        ET_count_embedding_300_15_pipeline_max = Pipeline([
                        ("Embedding", embeddingvectorizer.EmbeddingCountVectorizer(self.model_300_15, 'max')),
                        ('clf', OneVsRestClassifier(ExtraTreesClassifier())),
                        ])

        ET_tfidf_embedding_300_15_pipeline_max = Pipeline([
                        ("Embedding", embeddingvectorizer.EmbeddingTfidfVectorizer(self.model_300_15, 'max')),
                        ('clf', OneVsRestClassifier(ExtraTreesClassifier())),
                        ])


        ET_count_embedding_300_10_pipeline = Pipeline([
                        ("Embedding", embeddingvectorizer.EmbeddingCountVectorizer(self.model_300_10, 'mean')),
                        ('clf', OneVsRestClassifier(ExtraTreesClassifier())),
                        ])

        ET_tfidf_embedding_300_10_pipeline = Pipeline([
                        ("Embedding", embeddingvectorizer.EmbeddingTfidfVectorizer(self.model_300_10, 'mean')),
                        ('clf', OneVsRestClassifier(ExtraTreesClassifier())),
                        ])


        ET_count_embedding_300_10_pipeline_sum = Pipeline([
                        ("Embedding", embeddingvectorizer.EmbeddingCountVectorizer(self.model_300_10, 'sum')),
                        ('clf', OneVsRestClassifier(ExtraTreesClassifier())),
                        ])

        ET_tfidf_embedding_300_10_pipeline_sum = Pipeline([
                        ("Embedding", embeddingvectorizer.EmbeddingTfidfVectorizer(self.model_300_10, 'sum')),
                        ('clf', OneVsRestClassifier(ExtraTreesClassifier())),
                        ])

        ET_count_embedding_300_10_pipeline_max = Pipeline([
                        ("Embedding", embeddingvectorizer.EmbeddingCountVectorizer(self.model_300_10, 'max')),
                        ('clf', OneVsRestClassifier(ExtraTreesClassifier())),
                        ])

        ET_tfidf_embedding_300_10_pipeline_max = Pipeline([
                        ("Embedding", embeddingvectorizer.EmbeddingTfidfVectorizer(self.model_300_10, 'max')),
                        ('clf', OneVsRestClassifier(ExtraTreesClassifier())),
                        ])



        ET_count_embedding_100_10_pipeline = Pipeline([
                        ("Embedding", embeddingvectorizer.EmbeddingCountVectorizer(self.model_100_10, 'mean')),
                        ('clf', OneVsRestClassifier(ExtraTreesClassifier())),
                        ])

        ET_tfidf_embedding_100_10_pipeline = Pipeline([
                        ("Embedding", embeddingvectorizer.EmbeddingTfidfVectorizer(self.model_100_10, 'mean')),
                        ('clf', OneVsRestClassifier(ExtraTreesClassifier())),
                        ])


        ET_count_embedding_100_10_pipeline_sum = Pipeline([
                        ("Embedding", embeddingvectorizer.EmbeddingCountVectorizer(self.model_100_10, 'sum')),
                        ('clf', OneVsRestClassifier(ExtraTreesClassifier())),
                        ])

        ET_tfidf_embedding_100_10_pipeline_sum = Pipeline([
                        ("Embedding", embeddingvectorizer.EmbeddingTfidfVectorizer(self.model_100_10, 'sum')),
                        ('clf', OneVsRestClassifier(ExtraTreesClassifier())),
                        ])

        ET_count_embedding_100_10_pipeline_max = Pipeline([
                        ("Embedding", embeddingvectorizer.EmbeddingCountVectorizer(self.model_100_10, 'max')),
                        ('clf', OneVsRestClassifier(ExtraTreesClassifier())),
                        ])

        ET_tfidf_embedding_100_10_pipeline_max = Pipeline([
                        ("Embedding", embeddingvectorizer.EmbeddingTfidfVectorizer(self.model_100_10, 'max')),
                        ('clf', OneVsRestClassifier(ExtraTreesClassifier())),
                        ])




        ET_count_embedding_100_15_pipeline = Pipeline([
                        ("Embedding", embeddingvectorizer.EmbeddingCountVectorizer(self.model_100_15, 'mean')),
                        ('clf', OneVsRestClassifier(ExtraTreesClassifier())),
                        ])

        ET_tfidf_embedding_100_15_pipeline = Pipeline([
                        ("Embedding", embeddingvectorizer.EmbeddingTfidfVectorizer(self.model_100_15, 'mean')),
                        ('clf', OneVsRestClassifier(ExtraTreesClassifier())),
                        ])


        ET_count_embedding_100_15_pipeline_sum = Pipeline([
                        ("Embedding", embeddingvectorizer.EmbeddingCountVectorizer(self.model_100_15, 'sum')),
                        ('clf', OneVsRestClassifier(ExtraTreesClassifier())),
                        ])

        ET_tfidf_embedding_100_15_pipeline_sum = Pipeline([
                        ("Embedding", embeddingvectorizer.EmbeddingTfidfVectorizer(self.model_100_15, 'sum')),
                        ('clf', OneVsRestClassifier(ExtraTreesClassifier())),
                        ])

        ET_count_embedding_100_15_pipeline_max = Pipeline([
                        ("Embedding", embeddingvectorizer.EmbeddingCountVectorizer(self.model_100_15, 'max')),
                        ('clf', OneVsRestClassifier(ExtraTreesClassifier())),
                        ])

        ET_tfidf_embedding_100_15_pipeline_max = Pipeline([
                        ("Embedding", embeddingvectorizer.EmbeddingTfidfVectorizer(self.model_100_15, 'max')),
                        ('clf', OneVsRestClassifier(ExtraTreesClassifier())),
                        ])



        all_models = [
        ("SGD tfidf", SGD_tfidf_pipeline ) ,

        ("SGD count",  SGD_count_pipeline  ) ,

        ("SGD count embedding 300 15" ,  SGD_count_embedding_300_15_pipeline ) ,

        ("SGD tfidf embedding 300 15", SGD_tfidf_embedding_300_15_pipeline ) ,

        ("SGD count embedding sum 300 15",  SGD_count_embedding_300_15_pipeline_sum ) ,

        ("SGD tfidf embedding sum 300 15", SGD_tfidf_embedding_300_15_pipeline_sum ) ,

        ("SGD count embedding max 300 15",  SGD_count_embedding_300_15_pipeline_max ) ,

        ("SGD tfidf embedding max 300 15", SGD_tfidf_embedding_300_15_pipeline_max ) ,

        ("SGD count embedding 300 10" ,  SGD_count_embedding_300_10_pipeline ) ,

        ("SGD tfidf embedding 300 10", SGD_tfidf_embedding_300_10_pipeline ) ,

        ("SGD count embedding sum 300 10",  SGD_count_embedding_300_10_pipeline_sum ) ,

        ("SGD tfidf embedding sum 300 10", SGD_tfidf_embedding_300_10_pipeline_sum ) ,

        ("SGD count embedding max 300 10",  SGD_count_embedding_300_10_pipeline_max ) ,

        ("SGD tfidf embedding max 300 10", SGD_tfidf_embedding_300_10_pipeline_max ) ,

        ("SGD count embedding 100 10" ,  SGD_count_embedding_100_10_pipeline ) ,

        ("SGD tfidf embedding 100 10", SGD_tfidf_embedding_100_10_pipeline ) ,

        ("SGD count embedding sum 100 10",  SGD_count_embedding_100_10_pipeline_sum ) ,

        ("SGD tfidf embedding sum 100 10", SGD_tfidf_embedding_100_10_pipeline_sum ) ,

        ("SGD count embedding max 100 10",  SGD_count_embedding_100_10_pipeline_max ) ,

        ("SGD tfidf embedding max 100 10", SGD_tfidf_embedding_100_10_pipeline_max ) ,

        ("SGD count embedding 100 15" ,  SGD_count_embedding_100_15_pipeline ) ,

        ("SGD tfidf embedding 100 15", SGD_tfidf_embedding_100_15_pipeline ) ,

        ("SGD count embedding sum 100 15",  SGD_count_embedding_100_15_pipeline_sum ) ,

        ("SGD tfidf embedding sum 100 15", SGD_tfidf_embedding_100_15_pipeline_sum ) ,

        ("SGD count embedding max 100 15",  SGD_count_embedding_100_15_pipeline_max ) ,

        ("SGD tfidf embedding max 100 15", SGD_tfidf_embedding_100_15_pipeline_max ) ,

        ("SVC tfidf", SVC_tfidf_pipeline ) ,

        ("SVC count",  SVC_count_pipeline  ) ,


        ("SVC count embedding 300 15",  SVC_count_embedding_300_15_pipeline ) ,

        ("SVC tfidf embedding 300 15", SVC_tfidf_embedding_300_15_pipeline ) ,

        ("SVC count embedding sum 300 15",  SVC_count_embedding_300_15_pipeline_sum ) ,

        ("SVC tfidf embedding sum 300 15", SVC_tfidf_embedding_300_15_pipeline_sum ) ,

        ("SVC count embedding max 300 15",  SVC_count_embedding_300_15_pipeline_max ) ,

        ("SVC tfidf embedding max 300 15", SVC_tfidf_embedding_300_15_pipeline_max ) ,



        ("SVC count embedding 100 15",  SVC_count_embedding_100_15_pipeline ) ,

        ("SVC tfidf embedding 100 15", SVC_tfidf_embedding_100_15_pipeline ) ,

        ("SVC count embedding sum 100 15",  SVC_count_embedding_100_15_pipeline_sum ) ,

        ("SVC tfidf embedding sum 100 15", SVC_tfidf_embedding_100_15_pipeline_sum ) ,

        ("SVC count embedding max 100 15",  SVC_count_embedding_100_15_pipeline_max ) ,

        ("SVC tfidf embedding max 100 15", SVC_tfidf_embedding_100_15_pipeline_max ) ,


        ("SVC count embedding 100 10",  SVC_count_embedding_100_10_pipeline ) ,

        ("SVC tfidf embedding 100 10", SVC_tfidf_embedding_100_10_pipeline ) ,

        ("SVC count embedding sum 100 10",  SVC_count_embedding_100_10_pipeline_sum ) ,

        ("SVC tfidf embedding sum 100 10", SVC_tfidf_embedding_100_10_pipeline_sum ) ,

        ("SVC count embedding max 100 10",  SVC_count_embedding_100_10_pipeline_max ) ,

        ("SVC tfidf embedding max 100 10", SVC_tfidf_embedding_100_10_pipeline_max ) ,


        ("SVC count embedding 300 10",  SVC_count_embedding_300_10_pipeline ) ,

        ("SVC tfidf embedding 300 10", SVC_tfidf_embedding_300_10_pipeline ) ,

        ("SVC count embedding sum 300 10",  SVC_count_embedding_300_10_pipeline_sum ) ,

        ("SVC tfidf embedding sum 300 10", SVC_tfidf_embedding_300_10_pipeline_sum ) ,

        ("SVC count embedding max 300 10",  SVC_count_embedding_300_10_pipeline_max ) ,

        ("SVC tfidf embedding max 300 10", SVC_tfidf_embedding_300_10_pipeline_max ) ,



        ("PA tfidf", PA_tfidf_pipeline ) ,

        ("PA count", PA_count_pipeline ) ,


        ("PA count embedding 300 15", PA_count_embedding_300_15_pipeline  ) ,

        ("PA tfidf embedding 300 15", PA_tfidf_embedding_300_15_pipeline ) ,

        ("PA count embedding sum 300 15", PA_count_embedding_300_15_pipeline_sum  ) ,

        ("PA tfidf embedding sum 300 15", PA_tfidf_embedding_300_15_pipeline_sum ) ,

        ("PA count embedding max 300 15", PA_count_embedding_300_15_pipeline_max  ) ,

        ("PA tfidf embedding max 300 15", PA_tfidf_embedding_300_15_pipeline_max ) ,


        ("PA count embedding 100 15", PA_count_embedding_100_15_pipeline  ) ,

        ("PA tfidf embedding 100 15", PA_tfidf_embedding_100_15_pipeline ) ,

        ("PA count embedding sum 100 15", PA_count_embedding_100_15_pipeline_sum  ) ,

        ("PA tfidf embedding sum 100 15", PA_tfidf_embedding_100_15_pipeline_sum ) ,

        ("PA count embedding max 100 15", PA_count_embedding_100_15_pipeline_max  ) ,

        ("PA tfidf embedding max 100 15", PA_tfidf_embedding_100_15_pipeline_max ) ,


        ("PA count embedding 100 10", PA_count_embedding_100_10_pipeline  ) ,

        ("PA tfidf embedding 100 10", PA_tfidf_embedding_100_10_pipeline ) ,

        ("PA count embedding sum 100 10", PA_count_embedding_100_10_pipeline_sum  ) ,

        ("PA tfidf embedding sum 100 10", PA_tfidf_embedding_100_10_pipeline_sum ) ,

        ("PA count embedding max 100 10", PA_count_embedding_100_10_pipeline_max  ) ,

        ("PA tfidf embedding max 100 10", PA_tfidf_embedding_100_10_pipeline_max ) ,


        ("PA count embedding 300 10", PA_count_embedding_300_10_pipeline  ) ,

        ("PA tfidf embedding 300 10", PA_tfidf_embedding_300_10_pipeline ) ,

        ("PA count embedding sum 300 10", PA_count_embedding_300_10_pipeline_sum  ) ,

        ("PA tfidf embedding sum 300 10", PA_tfidf_embedding_300_10_pipeline_sum ) ,

        ("PA count embedding max 300 10", PA_count_embedding_300_10_pipeline_max  ) ,

        ("PA tfidf embedding max 300 10", PA_tfidf_embedding_300_10_pipeline_max ) ,



        ("ET tfidf", ET_tfidf_pipeline ) ,

        ("ET count", ET_count_pipeline ) ,


        ("ET count embedding 100 15", ET_count_embedding_100_15_pipeline ) ,

        ("ET tifdf embedding 100 15", ET_tfidf_embedding_100_15_pipeline ) ,

        ("ET count embedding sum 100 15", ET_count_embedding_100_15_pipeline_sum ) ,

        ("ET tifdf embedding sum 100 15", ET_tfidf_embedding_100_15_pipeline_sum ) ,

        ("ET count embedding max 100 15", ET_count_embedding_100_15_pipeline_max ) ,

        ("ET tifdf embedding max 100 15", ET_tfidf_embedding_100_15_pipeline_max ) ,


        ("ET count embedding 100 10", ET_count_embedding_100_10_pipeline ) ,

        ("ET tifdf embedding 100 10", ET_tfidf_embedding_100_10_pipeline ) ,

        ("ET count embedding sum 100 10", ET_count_embedding_100_10_pipeline_sum ) ,

        ("ET tifdf embedding sum 100 10", ET_tfidf_embedding_100_10_pipeline_sum ) ,

        ("ET count embedding max 100 10", ET_count_embedding_100_10_pipeline_max ) ,

        ("ET tifdf embedding max 100 10", ET_tfidf_embedding_100_10_pipeline_max ) ,


        ("ET count embedding 300 10", ET_count_embedding_300_10_pipeline ) ,

        ("ET tifdf embedding 300 10", ET_tfidf_embedding_300_10_pipeline ) ,

        ("ET count embedding sum 300 10", ET_count_embedding_300_10_pipeline_sum ) ,

        ("ET tifdf embedding sum 300 10", ET_tfidf_embedding_300_10_pipeline_sum ) ,

        ("ET count embedding max 300 10", ET_count_embedding_300_10_pipeline_max ) ,

        ("ET tifdf embedding max 300 10", ET_tfidf_embedding_300_10_pipeline_max ) ,


        ("ET count embedding 300 15", ET_count_embedding_300_15_pipeline ) ,

        ("ET tifdf embedding 300 15", ET_tfidf_embedding_300_15_pipeline ) ,

        ("ET count embedding sum 300 15", ET_count_embedding_300_15_pipeline_sum ) ,

        ("ET tifdf embedding sum 300 15", ET_tfidf_embedding_300_15_pipeline_sum ) ,

        ("ET count embedding max 300 15", ET_count_embedding_300_15_pipeline_max ) ,

        ("ET tifdf embedding max 300 15", ET_tfidf_embedding_300_15_pipeline_max )  ]


        return all_models


    def Benchmark(self, n):
        test_size = 1 - (n / float(len(self.train)))
        train_set, test_set = train_test_split(self.train, random_state=42, test_size=test_size)
        logging.info(f"new length of the training data: {len(train_set)}")
        return train_set, test_set

    def Clean_results(self):
        df = pd.DataFrame.from_dict([i for i in self.Get_results() ])
        df.sort_values(['algorithm', 'frame', 'length_training_set'])
        return df

    def Get_results(self):
        r = []
        f_results = []
        results = defaultdict(list)
        final_dict = {}

        for i in self.training_sizes:
            logging.info(f"TRAINING SIZE{i}\n\n\n\n")

            train_set, test_set = self.Benchmark(i)
            logging.info(f"Starting analyses with the a training set of the following size: {len(train_set)}")

            for f in self.frames:
                print(f)

                for name, model in self.Define_pipelines():
                    print(name)
                    clf = model.fit(train_set.text, train_set[f])
                    y_pred = clf.predict(self.test.text)

                    final_dict ={"length_training_set" : len(train_set) ,
                                 "frame" : f ,
                                 "algorithm" : name ,
                                 "accuracy" : accuracy_score(self.test[f], y_pred) ,
                                 "f1_weighted" : f1_score(self.test[f], y_pred, average='weighted') ,
                                 "f1_macro" : f1_score(self.test[f], y_pred, average='macro') ,
                                 "f1_micro" : f1_score(self.test[f], y_pred, average='micro') ,
                                 "recall_score_weighted" : recall_score(self.test[f], y_pred, average='weighted') ,
                                 "precision_weighted" : precision_score(self.test[f], y_pred, average='weighted') ,
                                 "recall_score_macro" : recall_score(self.test[f], y_pred, average='macro') ,
                                 "precision_macro" : precision_score(self.test[f], y_pred, average='macro') ,
                                 "recall_score_micro" : recall_score(self.test[f], y_pred, average='micro') ,
                                 "precision_micro" : precision_score(self.test[f], y_pred, average='micro') ,
                                  }

                    print(f"this is the final dict: {final_dict}")

                    r.append(final_dict)
            #f_results.append(r)
        return r

if __name__ == '__main__':

    a = Learning_rate_SML(outputpath='/home/anne/RPA-data/output/frames/', datapath='../data/intermediate/RPA_data_with_dictionaryscores.pkl',
                     path_to_embeddings = '/home/anne/RPA-data/output/')
    results = a.Clean_results()
    results.to_csv("/home/anne/RPA-data/output/frames/learning_rate_SML.csv")
    logging.info("Saved file as: /home/anne/RPA-data/output/frames/learning_rate_SML.csv")
