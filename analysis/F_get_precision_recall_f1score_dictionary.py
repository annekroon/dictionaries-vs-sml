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

OUTPUT_PATH ='../output/frames/'
PATH = '/Users/anne/surfdrive/uva/projects/RPA_KeepingScore/data/RPA_data_with_dictionaryscores.pkl'
PE = '/Users/anne/repos/embedding_models/RPA/w2v_300d2000-01-01_2018-12-31'

DICT_LENGTH = 30

df = pd.read_pickle(PATH)

# only keep newspaper data
df = df[df['type'] == 'newspaper']
frames = ['attrresp', 'cnflct','ecnmc', 'hmnintrst']

# 2 = not present, set to zero (0 = not present, 1 = present)
df[frames] = df[frames].replace({2:0})
df['attrresp'].fillna(0, inplace=True)
df.rename(columns= {'text_x': 'text'}, inplace=True)

train, test = train_test_split(df, random_state=42, test_size=0.3, shuffle=True)

x_train = train.text
x_test = test.text

## clean att
mystopwords = stopwords.words('dutch')
extra_stop = [line.strip() for line in open('../stopwords/stopwords_NL.txt').readlines() if len(line)>1]
mystopwords = set(mystopwords + extra_stop)

def get_dictionaries(frame):
    att = " ".join(train[frame].dropna().to_list())
    body_of_text="".join([l for l in att.lower() if l not in punctuation])
    body_of_text=" ".join(body_of_text.split())
    text_clean = " ".join([w for w in body_of_text.split() if w not in mystopwords and len(w) > 1])
    attribution_clean = " ".join([w for w in text_clean.split() if w.isalpha()])
    att = set(attribution_clean.split())
    return attribution_clean, att

logging.info("Some info on the dictionaries:\n\n")
logging.info("The dictionary Attribution of Responsibility contains {} words. These are the most common words:\n\n{}\n\n".format(len(get_dictionaries('attrresp_wrds')[1]), Counter(get_dictionaries('attrresp_wrds')[0].split()).most_common(5)))
logging.info("The dictionary Human Interst contains {} words. These are the most common words:\n\n{}\n\n".format(len(get_dictionaries('hmnintrst_wrds')[1]), Counter(get_dictionaries('hmnintrst_wrds')[0].split()).most_common(5)))
logging.info("The dictionary Economic Consequences contains {} words. These are the most common words:\n\n{}\n\n".format(len(get_dictionaries('ecnmc_wrds')[1]), Counter(get_dictionaries('ecnmc_wrds')[0].split()).most_common(5)))
logging.info("The dictionary Conflict contains {} words. These are the most common words:\n\n{}\n\n".format(len(get_dictionaries('cnflct_wrds')[1]), Counter(get_dictionaries('cnflct_wrds')[0].split()).most_common(5)))

def get_final_dicts():
    frame_dict = defaultdict()
    frame_dict['att_d'] = [i[0] for i in Counter(get_dictionaries('attrresp_wrds')[0].split()).most_common(DICT_LENGTH)]
    frame_dict['hmninstr_d'] = [i[0] for i in Counter(get_dictionaries('hmnintrst_wrds')[0].split()).most_common(DICT_LENGTH)]
    frame_dict['cnflct_d'] = [i[0] for i in Counter(get_dictionaries('cnflct_wrds')[0].split()).most_common(DICT_LENGTH)]
    frame_dict['ecnm_d'] = [i[0] for i in Counter(get_dictionaries('ecnmc_wrds')[0].split()).most_common(DICT_LENGTH)]
    return frame_dict

def get_stemmed_dict():
    stemmer = SnowballStemmer("dutch")
    d = get_final_dicts()
    stemmed_dictionary = {}
    for frame, words in d.items():
        stemmed_dictionary[frame] = [ stemmer.stem(w) for w in words ]
    return stemmed_dictionary

def map_dict_to_text(type_of_text, stemmed):
    result = []
    for document, documentnr in zip(test[type_of_text], test['documentnr']):
        topics_per_document = {}
        document = str(document)
        if stemmed == False:
            d = get_final_dicts()
        elif stemmed == True:
            d = get_stemmed_dict()
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
    df2 = pd.DataFrame.from_dict(result)
    df3 = df2.pivot(index='documentnr', columns='frame', values='len matches')
    df3[df3>1] = 1
    return df3

def get_tp_fp_fn(type_of_text, stemmed):

    '''create columns with true postives, false positives, and false negatives'''
  #  df = pd.merge(test, df3, how= 'left', on = 'documentnr')
    df3 = map_dict_to_text(type_of_text = type_of_text, stemmed = stemmed)
    df = pd.merge(test, df3, how= 'left', on = 'documentnr')
    frames = ['attrresp', 'cnflct','ecnmc', 'hmnintrst']
    frames_d = ['att_d' , 'cnflct_d', 'ecnm_d','hmninstr_d']


    for frame, frame_d in zip(frames, frames_d):
        columnname_tp = "_tp " + str(frame)
        columnname_fp = "_fp " + str(frame)
        columnname_fn = "_fn " + str(frame)
        columnname_tn = "_tn " + str(frame)

        df[columnname_tp] = np.where( (df[frame] == 1) & (df[frame_d] == 1) , 1, 0 )
            # false positive = dictionary identified, but golden standard not.
        df[columnname_fp] = np.where( (df[frame] != 1) & (df[frame_d] == 1) , 1, 0 )
        # false negative = dictionary NOT identified, but golden standard DID identify
        df[columnname_fn] = np.where( (df[frame] == 1) & (df[frame_d] != 1) , 1, 0 )
        df[columnname_tn] = np.where( (df[frame] == 0) & (df[frame_d] == 0) , 1, 0 )

    return df

def get_recall_precision(frames, type_of_text, stemmed):

    true_positives = ["_tp " + str(i) for i in frames]
    true_negatives = ["_tn " + str(i) for i in frames]
    false_positives = ["_fp " + str(i) for i in frames]
    false_negatives = ["_fn " + str(i) for i in frames]

    recall = {}
    precision = {}
    f1score = {}
    accuracy = {}

    df = get_tp_fp_fn(type_of_text = type_of_text, stemmed=stemmed)
    frames_d = ['att_d' , 'cnflct_d', 'ecnm_d','hmninstr_d']

    for tp, tn, fp, fn, frame, frame_d in zip(true_positives, true_negatives, false_positives, false_negatives, frames, frames_d) :

        #recall[str(frame)] = df[tp].sum(axis=0) / ( df[tp].sum(axis=0) + df[fn].sum(axis=0) )
        recall[str(frame)] = recall_score(df[frame], df[frame_d], average='macro', sample_weight=None)
        precision[str(frame)] = precision_score(df[frame], df[frame_d], average='macro', sample_weight=None)
        f1score[str(frame)] = f1_score(df[frame], df[frame_d], average='macro', sample_weight=None)


    #    precision[str(frame)] = df[tp].sum(axis=0) / ( df[tp].sum(axis=0) + df[fp].sum(axis=0) )
#        f1score[str(frame)] = 2 * ( ( precision[str(frame)] * recall[str(frame)] ) / ( precision[str(frame)] + recall[str(frame)] ) )
        accuracy[str(frame)] = (df[tp].sum(axis=0) + df[tn].sum(axis=0)) / (df[tp].sum(axis=0) + df[tn].sum(axis=0) + df[fp].sum(axis=0) + df[fn].sum(axis=0) )

    return recall, precision, f1score, accuracy

frames = ['attrresp', 'cnflct','ecnmc', 'hmnintrst']
recall, precision, f1score, accuracy = get_recall_precision(frames = frames, type_of_text = 'stemmed_text', stemmed=True)
total = { k: [ precision[k] , recall[k], f1score[k], accuracy[k]] for k in recall }

fname = '{}recision_recall_f1score_dictionary_stemmed_FRAMES.json'.format(OUTPUT_PATH)
logging.info("STEMMED: Recall: {}, precision: {}, f1_score: {}".format(recall, precision, f1score))

with open(fname, mode='w') as fo:
    json.dump(total, fo)

logging.info("Created file {}".format(fname))

recall, precision, f1score, accuracy = get_recall_precision(frames = frames, type_of_text = 'text', stemmed=False)
total = { k: [ precision[k] , recall[k], f1score[k], accuracy[k]] for k in recall }

fname = '{}recision_recall_f1score_dictionary_not_stemmed_FRAMES.json'.format(OUTPUT_PATH)
logging.info("NOT STEMMED: Recall: {}, precision: {}, f1_score: {}".format(recall, precision, f1score))

with open(fname, mode='w') as fo:
    json.dump(total, fo)

logging.info("Created file {}".format(fname))
