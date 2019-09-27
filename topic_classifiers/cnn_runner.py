import tensorflow as tf
import numpy as np
from tqdm import tqdm
import pandas as pd
import pickle
import numpy as np
import spacy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Embedding, LSTM
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
#from keras.preprocessing.sequence import pad_sequences
import string
from sklearn.metrics import classification_report

from tensorflow.python.keras.callbacks import Callback
from tensorflow.keras.metrics import Recall

import csv

import logging

from collections import defaultdict
import json

logger = logging.getLogger()
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
logging.root.setLevel(level=logging.INFO)

MODEL = "../../tmpanne/RPA/w2v_models/w2v_300d2000-01-01_2018-12-31.txt"
PATH_TO_DATA = "../../tmpanne/RPA"
DATA = "RPA_and_Buschers_data_with_dictionaryscores.pkl"
EPOCHS = 3

class Metrics(Callback):
    def __init__(self, x, y):
        self.x = x
        self.y = y if (y.ndim == 1 or y.shape[1] == 1) else np.argmax(y, axis=1)
        self.reports = []

    def on_epoch_end(self, epoch, logs={}):
        y_hat = np.asarray(self.model.predict(self.x))
        y_hat = np.where(y_hat > 0.5, 1, 0) if (y.ndim == 1 or y_hat.shape[1] == 1)  else np.argmax(y_hat, axis=1)
        report = classification_report(self.y,y_hat,output_dict=True)
        self.reports.append(report)
        return

    # Utility method
    def get(self, metrics, of_class):
        return [report[str(of_class)][metrics] for report in self.reports]

#metrics_milticlass = Metrics(Xtrain, y_train)
#milticlass_model.fit(x, y, epochs=30, callbacks=[metrics_milticlass])

class CNN_runner():
    '''This prepares a CNN model and runs it'''

    def __init__(self, path_to_embeddings, path_to_data, EPOCHS):
        self.nmodel = 0
        self.path_to_embeddings = path_to_embeddings
        self.path_to_data = path_to_data
        self.df = self.get_data()
        self.X_train = self.x_y_split() [0]
        self.X_test =  self.x_y_split() [1]
        self.y_train = self.x_y_split() [2]
        self.y_test =  self.x_y_split() [3]
        self.tokenizer = self.get_tok(self.X_train)
        self.epochs = EPOCHS

    def get_data(self):
        self.df = pd.read_pickle(self.path_to_data + DATA)
        self.df['main_topic_id'] = self.df['main_topic_label'].factorize()[0]
        return self.df

    def encodeY(self, Y):
        '''create one-hot (dummies) for output, see also https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/
        encode class values as integers
        '''
        encoder = LabelEncoder()
        encoder.fit(Y)
        encoded_Y = encoder.transform(Y)
        dummy_y = tf.keras.utils.to_categorical(encoded_Y)
        return dummy_y

    def x_y_split(self):
        X_train, X_test, y_train, y_test = train_test_split([t.translate(str.maketrans('', '', string.punctuation)) for t in self.df['text_x']], self.encodeY(self.df['main_topic_id'].map(int)), random_state = 42, test_size = 0.2)
        return X_train, X_test, y_train, y_test

    def get_tok(self, X_train):
    # create the tokenizer
        tokenizer = Tokenizer()
        # fit the tokenizer on the documents
        tokenizer.fit_on_texts(X_train)
        # sequence encode
        return tokenizer

    def get_embedding_index(self):
        embeddings_index = {}
        with open(self.path_to_embeddings) as f:
            numberofwordvectors, dimensions = [int(e) for e in next(f).split()]
            for line in tqdm(f):
                values = line.split()
                embeddings_index[values[0]] = np.asarray(values[1:], dtype='float32')
               # word = values[0]
               # coefs = np.asarray(values[1:], dtype='float32')
              #  embeddings_index[word] = coefs

        print('Found %s word vectors.' % len(embeddings_index))
        print('Should be {} vectors with {} dimensions'.format(numberofwordvectors, dimensions))
        return embeddings_index

    def get_weight_matrix(self, embedding, vocab):
        words_not_found = 0
        total_words = 0
        DEBUG_lijstmetwoorden = []
        # total vocabulary size plus 0 for unknown words
        vocab_size = len(vocab) + 1
        print("This is the length of the vocab", len(vocab))
        # define weight matrix dimensions with all 0
        weight_matrix = np.zeros((vocab_size, 300))
        # step vocab, store vectors using the Tokenizer's integer mapping
        for word, i in tqdm(vocab.items()):
            e = embedding.get(word, None)
            if e is not None:   # if we do not find the word, we do not want to replace anything but leave the zero's
                weight_matrix[i] = e
                total_words+=1
            else:
                words_not_found+=1
                DEBUG_lijstmetwoorden.append(word)
        print('Weight matrix created. For {} out of {} words, we did not have any embedding.'.format(words_not_found, total_words))
        return DEBUG_lijstmetwoorden, weight_matrix

    def create_embedding_layer(self):
        tokenizer = self.get_tok(self.X_train)
        embeddings_index = self.get_embedding_index()

        max_length = max([len(s.split()) for s in self.X_train])

        missingwords, embedding_vectors = self.get_weight_matrix(embeddings_index, tokenizer.word_index)
        logger.info("Embedding shape: {}".format(embedding_vectors.shape))
        logger.info("Length Embedding Vectors: {}".format(len(embedding_vectors)))

        embedding_layer = Embedding(len(tokenizer.word_index)+1, 300, weights=[embedding_vectors], input_length=max_length, trainable=False)
        logger.info("Created embedding layer...")
        return embedding_layer

    def getX(self):
        encoded_docs = self.tokenizer.texts_to_sequences(self.X_train)
        # pad sequences
        max_length = max([len(s.split()) for s in self.X_train])
        Xtrain = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
        # HIER OVER NADENKEN
        encoded_docs = self.tokenizer.texts_to_sequences(self.X_test)
        # pad sequences
        Xtest = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
        return Xtrain, Xtest

    def define_CNN_model(self):
        numberoflabels = len(self.df['main_topic_id'].unique())

        model = Sequential()
        model.add(self.create_embedding_layer()) # call embedding layer
        model.add(Conv1D(128, 4, activation='relu'))
        model.add(MaxPooling1D(4))
        model.add(MaxPooling1D(4))
        model.add(Flatten())
        model.add(Dense(units=64, activation='relu'))
        model.add(Dense(units=numberoflabels, activation='softmax'))   # voor twee categorien sigmoid, voor 1 tanh
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
        logger.info("Model created. Summary:\n\n{}".format(model.summary()) )
        return model

    def get_XandY(self):
        Xtrain, Xtest = self.getX()
        x = [t.translate(str.maketrans('', '', string.punctuation)) for t in self.df['text_x']]
        y = self.encodeY(self.df['main_topic_id'].map(int))
        return Xtrain, Xtest, self.y_train, self.y_test, x, y, self.df

mycnn = CNN_runner(MODEL, PATH_TO_DATA, EPOCHS = EPOCHS)
model = mycnn.define_CNN_model()
Xtrain, Xtest, y_train, y_test, x, y, df = mycnn.get_XandY()

metrics_milticlass = Metrics(Xtrain, y_train)
model.fit(Xtrain, y_train, epochs=EPOCHS, validation_data=[Xtest, y_test], verbose=True, callbacks=[metrics_milticlass])
fname = '{}cnnmodel_{}_epochs.h5'.format(PATH_TO_DATA, EPOCHS)
model.save(fname)


y_prob = model.predict(Xtest)
y_classes = y_prob.argmax(axis=-1)

#assert len(y_classes) == len(y_test.argmax(axis=-1))

predicted = list(y_classes)
actual = list(y_test.argmax(axis=-1))

save_actual_predicted = '{}predicted_actual.csv'.format(PATH_TO_DATA)

with open(save_actual_predicted, 'w') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerows(zip(predicted,actual))


    df['main_topic_id']  = df['main_topic_label'].factorize()[0]
labels = df.groupby('main_topic_id')['main_topic_label'].max().to_list()

d_f1 = defaultdict(list)

for m in ['f1-score']:
    for l in labels:
        d_f1[l] += metrics_milticlass.get(m,c)

d_recall = defaultdict(list)

for m in ['recall']:
    for l in labels:
        d_recall[l] += metrics_milticlass.get(m,c)

d_precision = defaultdict(list)

for m in ['precision']:
    for l in labels:
        d_precision[l] += metrics_milticlass.get(m,c)


f_f1 = '{}f1_scores.json'.format(PATH_TO_DATA)
f_re = '{}recall_scores.json'.format(PATH_TO_DATA)
f_pr = '{}precision_scores.json'.format(PATH_TO_DATA)

with open(f_f1, 'w') as fp:
    json.dump(d_f1, fp)

with open(f_re, 'w') as fp:
    json.dump(d_recall, fp)

from collections import defaultdict
import json


df = pd.read_pickle(PATH_TO_DATA + DATA)
df['main_topic_id']  = df['main_topic_label'].factorize()[0]
labels = df.groupby('main_topic_id')['main_topic_label'].max().to_list()

d_f1 = defaultdict(list)

for m in ['f1-score']:
    for l in labels:
        d_f1[l] += metrics_milticlass.get(m,c)

d_recall = defaultdict(list)

for m in ['recall']:
    for l in labels:
        d_recall[l] += metrics_milticlass.get(m,c)

d_precision = defaultdict(list)

for m in ['precision']:
    for l in labels:
        d_precision[l] += metrics_milticlass.get(m,c)

f_f1 = '{}f1_scores.json'.format(PATH_TO_DATA)
f_re = '{}recall_scores.json'.format(PATH_TO_DATA)
f_pr = '{}precision_scores.json'.format(PATH_TO_DATA)

with open(f_f1, 'w') as fp:
    json.dump(d_f1, fp)

with open(f_pr, 'w') as fp:
    json.dump(d_precision, fp)

with open(f_re, 'w') as fp:
    json.dump(d_recall, fp)
