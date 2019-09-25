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

PATH_TO_DATA = '~/surfdrive/uva/projects/RPA_KeepingScore/data/'
FILENAME = 'RPA_and_Buschers_data_with_dictionaryscores.pkl'

OUTPUT_PATH ='../output/'

def get_data():
    df = pd.read_pickle(PATH_TO_DATA + FILENAME)
    return df

def gridsearch_with_classifiers(sample):

    df = get_data()

    logging.info('getting the data. keeping sample: {}'.format(sample))

    if sample == 'totalsample':
        df = df
    elif sample == 'newspaper_sample_only':
        df = df[df['type'] == 'newspaper']
    elif sample == 'pq_sample_only' :
        df = df[df['type'] == 'parlementary question']
    elif sample == 'RPA_sample' :
        df = df[df['origin'] == 'RPA']
    elif sample == 'Bjorns_sample' :
        df = df[df['origin'] == 'Bjorn']

    logging.info('total size df: {}'.format(len(df)))
    X_train , X_test , y_train , y_test = train_test_split (df['text_x'], df['main_topic_label'], test_size = 0.2 , random_state =42)

    class_report = []
    results = []

    names = [
             "Naive Bayes",
             "Passive Agressive",
             "SGDClassifier"
            ]

    classifiers = [
        MultinomialNB(),
        PassiveAggressiveClassifier(),
        SGDClassifier(),
    ]

    parameters = [
                 {'vect__ngram_range': [(1, 1), (1, 2)],
                  'clf__alpha': (1e-2, 1e-3, 1e-5)},

                {

                'clf__loss': ('hinge', 'squared_hinge'),
                'clf__C': (0.01, 0.5, 1.0)   ,
                'clf__fit_intercept': (True, False) ,
                'vect__ngram_range': [(1, 1), (1, 2)] ,
                'tfidf__use_idf' :(True ,False),
                'clf__max_iter': (5 ,10 ,15)

                } ,

                  {'clf__max_iter': (20, 30) ,
                   'clf__alpha': (1e-2, 1e-3, 1e-5),
                   'clf__penalty': ('l2', 'elasticnet')}
                 ]


    for name, classifier, params in zip(names, classifiers, parameters):
        my_dict = {}
        print(name)
        print(classifier)
        print(params)
        clf_pipe = Pipeline([
            ('vect', TfidfVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', classifier),
        ])

        gs_clf = GridSearchCV(clf_pipe, param_grid=params, n_jobs=-1, cv=3)
        logger.info("Starting gridsearch....")
        clf = gs_clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)

        print("{} score: {}".format(name, score))
        print("{} are the best estimators".format(clf.best_estimator_))

        results_to_dict = classification_report((clf.best_estimator_.predict(X_test)), y_test, output_dict= True)

        results_to_dict['classifier:'] = classifier
        results_to_dict['best estimators:'] = clf.best_estimator_

        print("Created dictionary with classification report: \n\n{}".format(results_to_dict))

        class_report.append(results_to_dict)

        y_hats = clf.predict(X_test)

        my_dict = {"predicted": y_hats,
                     "actual" : y_test.values  ,
                     "classifier" : name}

        results.append(my_dict)

    return class_report, results


def get_scores(sample):
    class_report, results = gridsearch_with_classifiers(sample)
    fname_accuracy = '{}SML_precision_recall_f1score{}.json'.format(OUTPUT_PATH, sample)

    fname_predictions = '{}SML_predicted_actual_{}.json'.format(OUTPUT_PATH, sample)

    data =  dict((key,d[key]) for d in class_report for key in d)
    output = pd.DataFrame.from_dict(data)

    output.to_json(fname_accuracy)

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

#class_report, results = get_scores(sample="pq_sample_only")
if __name__ == "__main__":

    logger = logging.getLogger()
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)

    get_scores(sample="totalsample")
    get_scores(sample="newspaper_sample_only")
    get_scores(sample="pq_sample_only")
    get_scores(sample="RPA_sample")
    get_scores(sample="Bjorns_sample")

   # results_to_dict = metrics.classification_report((clf.best_estimator_.predict(X_test), y_test), output_dict=True )
    #print(classification_report(, y_pred, target_names=target_names))
