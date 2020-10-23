#!/usr/bin/env python3

from nltk.corpus import stopwords
from string import punctuation
import nltk
import pandas as pd
from collections import Counter
from nltk.stem.snowball import SnowballStemmer
import logging
from sklearn.model_selection import train_test_split
from collections import defaultdict
import numpy as np
import json
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Learning_rate_dictionaries():
    def __init__(self, outputpath, datapath):
        self.stopwords = set(stopwords.words('dutch') + [line.strip() for line in open('../stopwords/stopwords_NL.txt').readlines() if len(line)>1] )
        self.outputpath =outputpath
        self.datapath = datapath
        self.frames = ['attrresp', 'cnflct','ecnmc', 'hmnintrst']
        self.frames_d = ['att_d' , 'cnflct_d', 'ecnm_d','hmninstr_d']
        self.df = self.Prep_df()
        self.train, self.test = train_test_split(self.df, random_state=42, test_size=0.2)
        self.DICT_LENGTH = 30
        self.training_sizes = list(range(10, 900, 10))

    def Prep_df(self):

        df = pd.read_pickle(self.datapath)
        df = df[df['type'] == 'newspaper'] # only keep newspaper data
        df[self.frames] = df[self.frames].replace({2:0})# 2 = not present, set to zero (0 = not present, 1 = present)
        df['attrresp'].fillna(0, inplace=True)
        df.rename(columns= {'text_x': 'text'}, inplace=True)

        return df

    def Get_dictionaries(self, frame, train_set):

        att = " ".join(train_set[frame].dropna().to_list())
        body_of_text="".join([l for l in att.lower() if l not in punctuation])
        body_of_text=" ".join(body_of_text.split())
        text_clean = " ".join([w for w in body_of_text.split() if w not in self.stopwords and len(w) > 1])
        attribution_clean = " ".join([w for w in text_clean.split() if w.isalpha()])
        att = set(attribution_clean.split())
        return attribution_clean, att

    def Get_final_dicts(self, train_set):

        logging.info("Some info on the dictionaries:\n\n")
        logging.info("The dictionary Attribution of Responsibility contains {} words. These are the most common words:\n\n{}\n\n".format(len(self.Get_dictionaries('attrresp_wrds', train_set)[1]), Counter(self.Get_dictionaries('attrresp_wrds', train_set)[0].split()).most_common(5)))
        logging.info("The dictionary Human Interst contains {} words. These are the most common words:\n\n{}\n\n".format(len(self.Get_dictionaries('hmnintrst_wrds', train_set)[1]), Counter(self.Get_dictionaries('hmnintrst_wrds', train_set)[0].split()).most_common(5)))
        logging.info("The dictionary Economic Consequences contains {} words. These are the most common words:\n\n{}\n\n".format(len(self.Get_dictionaries('ecnmc_wrds', train_set)[1]), Counter(self.Get_dictionaries('ecnmc_wrds', train_set)[0].split()).most_common(5)))
        logging.info("The dictionary Conflict contains {} words. These are the most common words:\n\n{}\n\n".format(len(self.Get_dictionaries('cnflct_wrds', train_set)[1]), Counter(self.Get_dictionaries('cnflct_wrds', train_set)[0].split()).most_common(5)))

        frame_dict = defaultdict()
        frame_dict['att_d'] = [i[0] for i in Counter(self.Get_dictionaries('attrresp_wrds', train_set)[0].split()).most_common(self.DICT_LENGTH)]
        frame_dict['hmninstr_d'] = [i[0] for i in Counter(self.Get_dictionaries('hmnintrst_wrds', train_set)[0].split()).most_common(self.DICT_LENGTH)]
        frame_dict['cnflct_d'] = [i[0] for i in Counter(self.Get_dictionaries('cnflct_wrds', train_set)[0].split()).most_common(self.DICT_LENGTH)]
        frame_dict['ecnm_d'] = [i[0] for i in Counter(self.Get_dictionaries('ecnmc_wrds', train_set)[0].split()).most_common(self.DICT_LENGTH)]
        return frame_dict

    def Get_stemmed_dict(self, train_set):

        stemmer = SnowballStemmer("dutch")
        d = self.Get_final_dicts(train_set)
        stemmed_dictionary = {}
        print(d)
        for frame, words in d.items():
            stemmed_dictionary[frame] = [ stemmer.stem(w) for w in words ]
        return stemmed_dictionary


    def Map_dict_to_text(self, stemmed, train_set):

        result = []

        if stemmed == True: type_of_text = 'stemmed_text'
        else: type_of_text = 'text'

        if stemmed == False:
            print(f"Getting dictionaries for the traingset with length: {len(train_set)}")
            d = self.Get_final_dicts(train_set)
        elif stemmed == True:
            print(f"Getting dictionaries for the traingset with length: {len(train_set)}")
            d = self.Get_stemmed_dict(train_set)

        for document, documentnr in zip(self.test[type_of_text], self.test['documentnr']):
            topics_per_document = {}
            document = str(document)

            for topic, words in d.items():
                try:
                    match = [x for x in words if x in document.lower().split(' ')]
                except:
                    print("HU", document)
                    match = []

                topics_per_document = {'documentnr' : documentnr,
                                        'frame': topic,
                                        'len matches' : len(match),
                                        'words matches' : match }

                result.append(topics_per_document)

        df = pd.DataFrame.from_dict(result)
        df = df.pivot(index='documentnr', columns='frame', values='len matches')
        df[df>1] = 1

        return df

    def Get_tp_fp_fn(self, stemmed, train_set):

        '''create columns with true postives, false positives, and false negatives'''

        df = self.Map_dict_to_text(stemmed, train_set)
        df = pd.merge(self.test, df, how= 'left', on = 'documentnr')

        for frame, frame_d in zip(self.frames, self.frames_d):
            columnname_tp = "_tp " + str(frame)
            columnname_fp = "_fp " + str(frame)
            columnname_fn = "_fn " + str(frame)
            columnname_tn = "_tn " + str(frame)

            df[columnname_tp] = np.where( (df[frame] == 1) & (df[frame_d] == 1) , 1, 0 ) # false positive = dictionary identified, but golden standard not.
            df[columnname_fp] = np.where( (df[frame] != 1) & (df[frame_d] == 1) , 1, 0 ) # false negative = dictionary NOT identified, but golden standard DID identify
            df[columnname_fn] = np.where( (df[frame] == 1) & (df[frame_d] != 1) , 1, 0 )
            df[columnname_tn] = np.where( (df[frame] == 0) & (df[frame_d] == 0) , 1, 0 )

        return df

    def Get_recall_precision(self, stemmed, train_set):

        true_positives = ["_tp " + str(i) for i in self.frames]
        true_negatives = ["_tn " + str(i) for i in self.frames]
        false_positives = ["_fp " + str(i) for i in self.frames]
        false_negatives = ["_fn " + str(i) for i in self.frames]

        recall = {}
        precision = {}
        f1score = {}
        accuracy = {}

        df = self.Get_tp_fp_fn(stemmed, train_set)

        for tp, tn, fp, fn, frame, frame_d in zip(true_positives, true_negatives, false_positives, false_negatives, self.frames, self.frames_d) :
            recall[str(frame)] = recall_score(df[frame], df[frame_d], average='weighted', sample_weight=None)
            precision[str(frame)] = precision_score(df[frame], df[frame_d], average='weighted', sample_weight=None)
            f1score[str(frame)] = f1_score(df[frame], df[frame_d], average='weighted', sample_weight=None)
            accuracy[str(frame)] = (df[tp].sum(axis=0) + df[tn].sum(axis=0)) / (df[tp].sum(axis=0) + df[tn].sum(axis=0) + df[fp].sum(axis=0) + df[fn].sum(axis=0) )

        return recall, precision, f1score, accuracy

    def Benchmark(self, n):
        test_size = 1 - (n / float(len(self.train)))
        train_set, test_set = train_test_split(self.train, random_state=42, test_size=test_size)
        logging.info(f"new length of the training data: {len(train_set)}")
        return train_set, test_set

    def Get_results(self):
        final_results_stemmed = []
        final_results_not_stemmed = []

        for i in self.training_sizes:
            logging.info(f"TRAINING SIZE{i}\n\n\n\n")

            trains, tests = self.Benchmark(i)

           # df = self.Map_dict_to_text(stemmed=True, test_set=tests, train_set=trains)

            recall, precision, f1score, accuracy = self.Get_recall_precision(stemmed=True, train_set=trains)
            total = { k: [ precision[k] , recall[k], f1score[k], accuracy[k]] for k in recall }
            #total['trainingsize'] = [i] * 4

            #final_results_stemmed.append(total)
            final_results_stemmed.append({len(trains): total})
            logging.info(f"STEMMED: Recall: {recall}, precision: {precision}, f1_score: {f1score}")

            recall, precision, f1score, accuracy = self.Get_recall_precision(stemmed=False, train_set=trains)
            total = { k: [ precision[k] , recall[k], f1score[k], accuracy[k]] for k in recall }
            #total['trainingsize'] = [i] * 4

            #final_results_not_stemmed.append(total)
            final_results_not_stemmed.append({len(trains): total})
            logging.info(f"NOT STEMMED: Recall: {recall}, precision: {precision}, f1_score: {f1score}")


        fname = '{}precision_recall_f1score_dictionary_stemmed_FRAMES.json'.format(self.outputpath)
        with open(fname, mode='w') as fo:
            json.dump(final_results_stemmed, fo)
            logging.info("Created file {}".format(fname))

        fname = '{}precision_recall_f1score_dictionary_not_stemmed_FRAMES.json'.format(self.outputpath)
        with open(fname, mode='w') as fo:
            json.dump(final_results_not_stemmed, fo)
            logging.info("Created file {}".format(fname))

if __name__ == '__main__':
    a = Learning_rate_dictionaries(outputpath='../output/frames/new/', datapath='../data/intermediate/RPA_data_with_dictionaryscores.pkl')
    a.Get_results()
