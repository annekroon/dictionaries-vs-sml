#!/usr/bin/env python3

import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer,  TfidfTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_iris
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
import logging
import json
from sklearn.svm import SVC
import embeddingvectorizer
from sklearn.ensemble import ExtraTreesClassifier
import gensim


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Learning_rate_SML_topics():

    def __init__(self, outputpath, datapath, path_to_embeddings, vect):
        self.vect = vect
        self.outputpath = outputpath
        self.data_path = datapath
        self.path_to_embeddings = path_to_embeddings
        self.model = self.Load_embedding_model()
        self.df = self.Get_data()
        self.X_train , self.X_test, self.y_train, self.y_test = train_test_split(self.df['text_clean'], self.df['main_topic_label'], test_size = 0.2 , random_state =0)
        self.training_sizes = [i for i in range(100, len(self.X_train), 200) ]
        self.names = ["Passive Agressive", "SGDClassifier" ,"SVM", "ET"]
        self.classifiers = [PassiveAggressiveClassifier(), SGDClassifier(), SVC(), ExtraTreesClassifier() ]
        self.parameters = [ {   'clf__loss': ('hinge', 'squared_hinge'),
                                'clf__C': (0.01, 0.5, 1.0)   ,
                                'clf__fit_intercept': (True, False) ,
                                'clf__max_iter': (5 ,10 ,15)} ,

                            {   'clf__max_iter': (20, 30) ,
                                'clf__alpha': (1e-2, 1e-3, 1e-5),
                                'clf__penalty': ('l2', 'elasticnet')} ,

                            {   'clf__C': [1, 10, 100, 1000],
                                'clf__gamma': [0.001, 0.0001],
                                'clf__kernel': ['rbf', 'linear']},

                            {  "clf__max_features": ['auto', 'sqrt', 'log2'] } ]

    def Get_data(self):
        df = pd.read_pickle(self.data_path)
        return df

    def Load_embedding_model(self):
        logging.info("\nLoading model...")
        mod = gensim.models.Word2Vec.load(self.path_to_embeddings)
        MDL = dict(zip(mod.wv.index2word, mod.wv.syn0))
        return MDL

    def Define_vectorizer(self):

        logging.info("this is length of the dataframe: {}".format(len(self.df)))

        if self.vect == 'tfidf':
            logging.info("the vectorizers is: {}".format(self.vect))
            VECT = TfidfVectorizer()

        elif self.vect == 'count':
            logging.info("the vectorizers is: {}".format(self.vect))
            VECT = CountVectorizer()

        elif self.vect == 'w2v_count':
            logging.info("the vectorizers is: {}".format(self.vect))

            VECT = embeddingvectorizer.EmbeddingCountVectorizer(self.model, 'mean')

        elif self.vect == 'w2v_tfidf':
            logging.info("the vectorizers is: {}".format(self.vect))
            VECT = embeddingvectorizer.EmbeddingTfidfVectorizer(self.model, 'mean')

        return VECT

    def Benchmark(self, n):
        test_size = 1 - (n / float(len(self.X_train)))

        X_train , X_test, y_train, y_test = train_test_split(self.X_train, self.y_train, test_size = test_size , random_state =0)
        logging.info(f"new length of the training TEXT: {len(X_train)} and LABELS {len(y_train)} ")
        assert len(X_train) == len(y_train)
        return X_train, y_train


    def Run_models(self):

        f_results_class = []
        f_true_predicted = []
        class_report = []
        results = []


        for trainingsize in self.training_sizes:
            logging.info(f"TRAINING SIZE{trainingsize}\n\n\n\n")

            X_train, y_train = self.Benchmark(trainingsize)
            logging.info(f"Starting analyses with the a training set of the following size: TEXT: {len(X_train)} LABELS: {len(y_train)}")


            for name, classifier, params in zip(self.names, self.classifiers, self.parameters):
                my_dict = {}
                logging.info(name)
                logging.info(classifier)
                logging.info(params)
                clf_pipe = Pipeline([
                    ('vect', self.Define_vectorizer()),
                    ('clf', classifier),
                ])


                gs_clf = GridSearchCV(clf_pipe, param_grid=params, cv=5)
                logger.info("Starting gridsearch....")
                clf = gs_clf.fit(X_train, y_train)
                score = clf.score(self.X_test, self.y_test)

                logging.info("{} score: {}".format(name, score))
                logging.info("{} are the best estimators".format(clf.best_estimator_))

                results_to_dict = classification_report((clf.best_estimator_.predict(self.X_test)), self.y_test, output_dict= True)

                results_to_dict['classifier:'] = name
                results_to_dict['training_size'] = trainingsize
                results_to_dict['best estimators:'] = clf.best_params_

                logging.info("Created dictionary with classification report: \n\n{}".format(results_to_dict))

                y_hats = clf.predict(self.X_test)

                my_dict = {"predicted": y_hats,
                           "actual" : self.y_test.values  ,
                           "classifier" : name,
                           "training_size": trainingsize}

                results.append(my_dict)
                class_report.append(results_to_dict)

        f_true_predicted.append(results)
        f_results_class.append(class_report)
        return f_results_class, f_true_predicted

    def Get_scores(self):

        class_report, results = self.Run_models()

        fname_accuracy = '{}SML_precision_recall_f1score_text_cleaned_{}.json'.format(self.outputpath, self.vect)
        fname_predictions = '{}SML_predicted_actual_text_cleaned_{}.json'.format(self.outputpath, self.vect)

        with open(fname_accuracy, mode = 'w') as fo:
            json.dump(class_report, fo)

        return class_report, results

        data = pd.DataFrame.from_dict([l for item in results for l in item ] )

        predicted = data.predicted.apply(pd.Series) \
                .merge(data, right_index = True, left_index = True) \
                .drop(["predicted"], axis = 1) \
                .melt(id_vars = ['classifier', 'training_size'], value_name = "Predicted label")
        actual = data.actual.apply(pd.Series) \
                .merge(data, right_index = True, left_index = True) \
                .drop(["predicted"], axis = 1) \
                .melt(id_vars = ['classifier', 'training_size'], value_name = "Actual label")

        df = pd.merge(predicted, actual, how = 'inner', left_index = True, right_index = True)

        df['Classifier'] = df['classifier_x']
        df['training_size'] = df['training_size_x']
        df = df[df.variable_x != 'actual']
        df = df[['Predicted label', 'Actual label', 'Classifier', 'training_size']]
        df.to_json(fname_predictions)


if __name__ == "__main__":

    logger = logging.getLogger()
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)

    for vectorizer in [ "w2v_tfidf"]:

        a = Learning_rate_SML_topics(outputpath='../output/topics/',
                                     datapath = '/Users/anne/repos/RPA/data/intermediate/RPA_data_with_dictionaryscores.pkl',
                                     vect = vectorizer,
                                     path_to_embeddings = '/home/anne/embedding_model/w2v_300d2000-01-01_2018-12-31')

        class_report, results = a.Get_scores()
