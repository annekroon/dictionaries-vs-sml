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


PATH_TO_DATA = '..data/intermediate/'
FILENAME = 'RPA_data_with_dictionaryscores.pkl'

OUTPUT_PATH ='../output/'

print("\nLoading model")

def get_data():
    df = pd.read_pickle(PATH_TO_DATA + FILENAME)
 #   df['main_topic_label'].replace({'Wetenschappelijk onderzoek, technologie en communicatie': 'Overige'}, inplace=True)
#    df = df[df['main_topic_label'].map(df.main_topic_label.value_counts()>190)]
  #  df = df[df['main_topic_label'].map(df.main_topic_label.value_counts()>150)]
#    df.main_topic_label.fillna(value='Overige', inplace=True)
    return df

def gridsearch_with_classifiers(sample, vect):

    df = get_data()
    print("this is length of the dataframe: {}".format(len(df)))
    logging.info('getting the data. keeping sample: {}'.format(sample))

    if sample == 'newspaper_sample_only':
        df = df[df['type'] == 'newspaper']
    elif sample == 'pq_sample_only' :
        df = df[df['type'] == 'parlementary question']
    elif sample == 'RPA_sample' :
        df = df[df['origin'] == 'RPA']

    if vect == 'tfidf':
        logging.info("the vectorizers is: {}".format(vect))
        VECT = TfidfVectorizer()

    elif vect == 'count':
        logging.info("the vectorizers is: {}".format(vect))
        VECT = CountVectorizer()

    elif vect == 'w2v_count':
        logging.info("the vectorizers is: {}".format(vect))

        PE = '/home/anne/tmpanne/RPA/w2v_models/w2v_300d2000-01-01_2018-12-31'
        mod = gensim.models.Word2Vec.load(PE)
        MDL = dict(zip(mod.wv.index2word, mod.wv.syn0))
        VECT = embeddingvectorizer.EmbeddingCountVectorizer(MDL, 'mean')

    elif vect == 'w2v_tfidf':
        logging.info("the vectorizers is: {}".format(vect))

        PE = '/home/anne/tmpanne/RPA/w2v_models/w2v_300d2000-01-01_2018-12-31'
        mod = gensim.models.Word2Vec.load(PE)
        MDL = dict(zip(mod.wv.index2word, mod.wv.syn0))
        VECT = embeddingvectorizer.EmbeddingTfidfVectorizer(MDL, 'mean')


    logging.info('total size df: {}'.format(len(df)))

    X_train , X_test , y_train , y_test = train_test_split (df['text_clean'], df['main_topic_label'], test_size = 0.2 , random_state =0)

    class_report = []
    results = []

    names = [
             "Passive Agressive",
             "SGDClassifier" ,
             "SVM",
             "ET"
            ]

    classifiers = [
        PassiveAggressiveClassifier(),
        SGDClassifier(),
        SVC(),
        ExtraTreesClassifier()
    ]

    parameters = [

                {

                'clf__loss': ('hinge', 'squared_hinge'),
                'clf__C': (0.01, 0.5, 1.0)   ,
                'clf__fit_intercept': (True, False) ,
                #'vect__ngram_range': [(1, 1), (1, 2)] ,
            #    'tfidf__use_idf' :(True ,False),
                'clf__max_iter': (5 ,10 ,15)

                } ,

                  {'clf__max_iter': (20, 30) ,
                   'clf__alpha': (1e-2, 1e-3, 1e-5),
                   'clf__penalty': ('l2', 'elasticnet')} ,

                   {'clf__C': [1, 10, 100, 1000],
                   'clf__gamma': [0.001, 0.0001],
                   'clf__kernel': ['rbf', 'linear']},


                   { "clf__max_features": ['auto', 'sqrt', 'log2'] }

                 ]


    for name, classifier, params in zip(names, classifiers, parameters):
        my_dict = {}
        print(name)
        print(classifier)
        print(params)
        clf_pipe = Pipeline([
            ('vect', VECT),
            ('clf', classifier),
        ])


        gs_clf = GridSearchCV(clf_pipe, param_grid=params, cv=5)
        logger.info("Starting gridsearch....")
        clf = gs_clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)

        print("{} score: {}".format(name, score))
        print("{} are the best estimators".format(clf.best_estimator_))

        results_to_dict = classification_report((clf.best_estimator_.predict(X_test)), y_test, output_dict= True)

        results_to_dict['classifier:'] = name
        results_to_dict['best estimators:'] = clf.best_params_

        print("Created dictionary with classification report: \n\n{}".format(results_to_dict))

        y_hats = clf.predict(X_test)

        my_dict = {"predicted": y_hats,
                   "actual" : y_test.values  ,
                    "classifier" : name}

        results.append(my_dict)
        class_report.append(results_to_dict)

    return class_report, results

def get_scores(sample, vect):
    class_report, results = gridsearch_with_classifiers(sample, vect)
    fname_accuracy = '{}SML_precision_recall_f1score_text_cleaned_{}_{}.json'.format(OUTPUT_PATH, sample, vect)
    fname_predictions = '{}SML_predicted_actual_text_cleaned_{}_{}.json'.format(OUTPUT_PATH, sample, vect)

    with open(fname_accuracy, mode = 'w') as fo:
        json.dump(class_report, fo)

    data = pd.DataFrame.from_dict(results)

    predicted = data.predicted.apply(pd.Series) \
        .merge(data, right_index = True, left_index = True) \
        .drop(["predicted"], axis = 1) \
        .melt(id_vars = ['classifier'], value_name = "Predicted label")

    actual = data.actual.apply(pd.Series) \
        .merge(data, right_index = True, left_index = True) \
        .drop(["predicted"], axis = 1) \
        .melt(id_vars = ['classifier'], value_name = "Actual label")

    df = pd.merge(predicted, actual, how = 'inner', left_index = True, right_index = True)

    df['Classifier'] = df['classifier_x']
    df = df[df.variable_x != 'actual']
    df = df[['Predicted label', 'Actual label', 'Classifier']]

    df.to_json(fname_predictions)

if __name__ == "__main__":

    logger = logging.getLogger()
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)

    get_scores(sample="newspaper_sample_only", vect = "count")
    get_scores(sample="pq_sample_only", vect = "count")
    get_scores(sample="RPA_sample", vect = "count")
#"w2v_count", "w2v_tfidf", "count", "tfidf"
