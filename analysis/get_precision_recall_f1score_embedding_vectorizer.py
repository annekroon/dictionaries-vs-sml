from sklearn.model_selection import GridSearchCV
import gensim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
import embeddingvectorizer
import json
import csv

OUTPUT_PATH ='../output/'
PATH = '/Users/anne/surfdrive/uva/projects/RPA_KeepingScore/data/RPA_data_with_dictionaryscores.pkl'
PE = '/Users/anne/repos/embedding_models/RPA/w2v_300d2000-01-01_2018-12-31'

print("\nLoading model")
mod = gensim.models.Word2Vec.load(PE)
MDL = dict(zip(mod.wv.index2word, mod.wv.syn0))

def run_classifier(sample):

    df = pd.read_pickle(PATH)
    if sample == 'totalsample':
        df = df
    elif sample == '':
        df = df[df['type'] == 'newspaper']
    elif sample == 'pq_sample_only' :
        df = df[df['type'] == 'parlementary question']
    elif sample == 'RPA_sample' :
        df = df[df['origin'] == 'RPA']
    elif sample == 'Bjorns_sample' :
        df = df[df['origin'] == 'Bjorn']
    X_train, X_test, y_train, y_test = train_test_split (df['text_clean'], df['main_topic_label'], test_size = 0.2, random_state= 42)

    ET_embedding = Pipeline([
    ("word2vec vectorizer", embeddingvectorizer.EmbeddingTfidfVectorizer(MDL, 'mean')),
    ("classifier", ExtraTreesClassifier(n_estimators=400))
    ])

    param_grid = { #"classifier__max_depth": [3, 1000],
#                "classifier__max_features": [1, 3, 10],
#                "classifier__criterion": ["gini", "entropy"] ,
#                "classifier__bootstrap": [True, False] ,
                "classifier__max_features": ['auto', 'sqrt', 'log2']}


    rf_grid = GridSearchCV(estimator=ET_embedding, param_grid=param_grid, verbose=10) #cv=10
    rf_detector = rf_grid.fit(X_train, y_train)

    #print('Best estimator.............:', rf_detector.best_estimator_.get_params() )

    print(classification_report(rf_detector.best_estimator_.predict(X_test), y_test) )
    results = classification_report(rf_detector.best_estimator_.predict(X_test), y_test, output_dict=True)

    fname_accuracy = '{}precision_recall_f1score_embedding_vectorizer_text_clean{}.json'.format(OUTPUT_PATH, sample)

    print(results)

    with open(fname_accuracy, mode = 'w') as fo:
        json.dump(results, fo)

    save_actual_predicted = '{}predicted_actual_embedding_vectorizer_text_clean{}.csv'.format(OUTPUT_PATH, sample)

    with open(save_actual_predicted, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(zip(rf_detector.best_estimator_.predict(X_test),y_test))

get_results = run_classifier("RPA_sample")
get_results = run_classifier("newspaper_sample_only")
get_results = run_classifier("pq_sample_only")
