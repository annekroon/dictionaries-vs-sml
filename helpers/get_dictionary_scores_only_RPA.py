import os, re, logging
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from collections import defaultdict
from nltk.stem.snowball import SnowballStemmer
import seaborn as sns
import logging
from nltk.corpus import stopwords
from string import punctuation
import nltk
import pandas as pd


BASE_DICT = '/Users/anne/repos/RPA/resources/'
FILENAME_DICT = '20140718_dutchdictionary.txt'
PATH_TO_DATA = '~/surfdrive/uva/projects/RPA_KeepingScore/data/'

MINNUMBERMATCHES = 2 # min number of times a keyword should occur for a topic to be present

stemmer = SnowballStemmer("dutch")

def label_topic(x):
    if x == '1':
        return 'Macro-economie en belastingen'
    if x == '2':
        return 'Burgerrechten en vrijheden'
    if x == '3':
        return 'Gezondheid'
    if x == '4':
        return 'Landbouw en Visserij'
    if x == '5':
        return 'Arbeid'
    if x == '6':
        return 'Onderwijs'
    if x == '7':
        return 'Milieu'
    if x == '8':
        return 'Energiebeleid'
    if x == '9':
        return 'Immigratie en integratie'
    if x == '10':
        return 'Verkeer en vervoer'
    if x == '11':
        return 'Unkown'
    if x == '12':
        return 'Justitie, Rechtspraak, Criminaliteit'
    if x == '13':
        return 'sociale Zaken'
    if x == '14':
        return 'Gemeenschapsontwikkeling, huisvestingsbeleid en stedelijke planning'
    if x == '15':
        return 'Ondernemingen, Bankwezen en binnenlandse handel '
    if x == '16':
        return 'Defensie'
    if x == '17':
        return 'Wetenschappelijk onderzoek, technologie en communicatie'
    if x == '18':
        return 'Buitenlandse handel'
    if x == '19':
        return 'Buitenlandse zaken en ontwikkelingssamenwerking'
    if x == '20':
        return 'Functioneren democratie en openbaar bestuur'
    if x == '21':
        return 'Ruimtelijke ordening, publiek natuur- en waterbeheer'
    if x == '22':
        return 'Unkown 2'
    if x == '23':
        return 'Kunst, cultuur en entertainment'
    if x == '24':
        return '*** Gemeentelijk en provinciaal bestuur'
    if x == '29':
        return '*** Sport'
    if x == '00':
        return 'Toegevoegde codes voor media'

def parse_xml():
    '''reads file with topic numbers + words and parses the title'''

    words = []
    topics = []
    for l in [line.strip() for line in open(os.path.join(BASE_DICT , FILENAME_DICT)).readlines() if len(line)>1] :
        topics_words = defaultdict(list)
        if l.startswith('<cnode'):
            wordlist = []
            topics_l = list(re.sub('">|"|t', '', l.split('=')[1]) )
            if len(topics_l) == 2 :
                final_topic = "".join(topics_l)
            elif len(topics_l) == 3 :
                final_topic = topics_l[0]
            elif len(topics_l) == 4 :
                final_topic = "".join( topics_l[:2] )
        elif l.startswith('<pnode'):
            word = re.sub('">|</pnode>|"', '', l.split('=')[1])
            words.append(word)
            topics.append(final_topic)
    return words, topics

def get_dict():
    'returns a dict with keys = topic, values = words '

    words, topics = parse_xml()
    d = defaultdict(list)
    for topic, word in zip(topics, words):
        topic_name = label_topic(topic)
        d[topic_name].append(word)
    return d

def get_stemmed_dict():
    stemmer = SnowballStemmer("dutch")
    d = get_dict()
    stemmed_dictionary = {}
    for topic, words in d.items():
        stemmed_dictionary[topic] = [ stemmer.stem(w) for w in words ]
    return stemmed_dictionary

def get_raw_data():
    columns_to_keep = ['text', 'topic', 'main_topic_label', 'attrresp', 'attrresp_wrds', 'hmnintrst', 'hmnintrst_wrds', 'cnflct', 'cnflct_wrds', 'ecnmc', 'ecnmc_wrds']

    df = pd.read_pickle(PATH_TO_DATA + 'VK_TEL_merged_with_annotated.pkl')
    df['text_title'] = df['text'].astype(str) + ' ' + df['title'].astype(str)
    del df['text']
    df.rename(columns={'text_title' : 'text', 'main_topic' : 'topic'}, inplace = True)
    df = df[columns_to_keep]
    df['type'] = 'newspaper'

    df2 = pd.read_pickle(PATH_TO_DATA + 'kamervragen_merged_with_annotated')
    df2.rename(columns={'questions' : 'text', 'main_topic' : 'topic'}, inplace = True)
    df2 = df2[columns_to_keep]
    df2['type'] = 'parlementary question'

    df = df.append(df2)

    df['origin'] = 'RPA'
    df.reset_index(drop=True, inplace=True)
    df['documentnr'] = df.index
    logger.info("Appended the kamervragen dataset to the newspaper dataset, resulting in a df with a len of {}".format(len(df)))
    return df

def get_data():
    df = get_raw_data()
    a = ['Gemeenschapsontwikkeling, huisvestingsbeleid en stedelijke planning' , 'Landbouw en Visserij', 'Macro-economie en belastingen', 'Wetenschappelijk onderzoek, technologie en communicatie',  'Toegevoegde codes voor media',  'Buitenlandse handel',  'Kunst, cultuur en entertainment', 'Energiebeleid', 'Ruimtelijke ordening, publiek natuur- en waterbeheer']
    b = ['Overige'] * len(a)
    overige_cat = dict(zip(a,b))
    df['main_topic_label'].replace(overige_cat, inplace = True)
    df.reset_index(drop=True, inplace=True)
    df['documentnr'] = df.index
    logger.info("Retrieved the recoded dataset containing {} cases".format(len(df)))

    body_of_text = df['text'].to_list()
    print(body_of_text[0][:501])

    body_of_text=["".join([l for l in speech if l not in punctuation]) for speech in body_of_text]  #remove punctuation
    body_of_text=[speech.lower() for speech in body_of_text]  # convert to lower case
    body_of_text=[" ".join(speech.split()) for speech in body_of_text]

    mystopwords = stopwords.words('dutch')
    extra_stop = [line.strip() for line in open('../stopwords/stopwords_NL.txt').readlines() if len(line)>1]
    mystopwords = set(mystopwords + extra_stop)
    text_clean = [" ".join([w for w in speech.split() if w not in mystopwords]) for speech in body_of_text]
    text_clean = [" ".join([w for w in speech.split() if w.isalpha()]) for speech in text_clean] # keep only words, no digits
    print("HERE COMES THE CLEANED TEXT: \n\n\n\n")
    print(text_clean[0][:501])
    df['text_clean'] = text_clean
    return df

def stem_sentences(sentence):
    try:
        tokens = sentence.split()
        stemmed_tokens = [stemmer.stem(token) for token in tokens]
        return ' '.join(stemmed_tokens)
    except:
        return 'NAN'

def return_stemmed_text_columns():
    df = get_data()
    logger.info("Start stemming....")
    df['stemmed_text'] = df.text_clean.apply(stem_sentences)
    return df

def dictionary_topics():
    df1 = return_stemmed_text_columns()
    result = []
    documentnr = -1
    logger.info("Start word search....")
    for document in df1['text_clean']:
        documentnr += 1
        topics_per_document = {}
        d = get_dict()
        for topic, words in d.items():
            match = [x for x in words if x in document.lower().split(' ')]
            doc_string = document.lower().split(' ')
            index = [doc_string.index(word) for word in match ]
            try:
                index_smallest = min(index)
            except:
                index_smallest = np.nan

            topics_per_document = {'documentnr' : documentnr,
                                    'topic_label_dictionary': topic,
                                    'index_words' : index,
                                    'smallest_index' : index_smallest,
                                    'len matches' : len(match),
                                    'words matches' : match  ,
                                    'text' : document.lower()}
            result.append(topics_per_document)
    return result

def dictionary_topics_stemmed():
    df1 = return_stemmed_text_columns()
    result = []
    documentnr = -1
    logger.info("Start stemmed word search....")
    for document in df1['stemmed_text']:
        documentnr += 1
        topics_per_document = {}
        d = get_stemmed_dict()
        for topic, words in d.items():
            match = [x for x in words if x in document.lower().split(' ')]
            doc_string = document.lower().split(' ')
            index = [doc_string.index(word) for word in match ]
            try:
                index_smallest = min(index)
            except:
                index_smallest = np.nan

            topics_per_document = {'documentnr' : documentnr,
                                    'stemmed_topic_label_dictionary': topic,
                                    'stemmed_index_words' : index,
                                    'stemmed_smallest_index' : index_smallest,
                                    'stemmed_len matches' : len(match),
                                    'stemmed_words matches' : match  ,
                                    'stemmed_text' : document.lower()}
            result.append(topics_per_document)
    return result

def get_merged_df():
    '''returns a df with number of topics as identified by the dictionary approach'''

    result = dictionary_topics()
    stemmed_results = dictionary_topics_stemmed()
    df2 = pd.DataFrame.from_dict(result)
    df2 = (df2.assign(to_sort = df2.smallest_index.abs()).sort_values('to_sort').drop_duplicates('documentnr').drop(columns='to_sort'))
    df2 = df2[np.isfinite(df2['smallest_index'])]
    df3 = pd.DataFrame.from_dict(stemmed_results)
    df3 = (df3.assign(to_sort = df3.stemmed_smallest_index.abs()).sort_values('to_sort').drop_duplicates('documentnr').drop(columns='to_sort'))
    df3 = df3[np.isfinite(df3['stemmed_smallest_index'])]
    df1 = get_data()
    df = pd.merge(df1, df2, how= 'left', on = 'documentnr')
    df = pd.merge(df, df3, how = 'left', on='documentnr')
    df['topic_label_dictionary'].fillna(value='Overige', inplace = True)
    df['len matches'] = df['len matches'].fillna(0)
    df['stemmed_topic_label_dictionary'].fillna(value='Overige', inplace = True)
    df['stemmed_len matches'] = df['stemmed_len matches'].fillna(0)
    return df

def recode_dictionary():
    '''recode categories so to match Bjorns' scoring'''

    df = get_merged_df()
    a = ['Gemeenschapsontwikkeling, huisvestingsbeleid en stedelijke planning' , 'Landbouw en Visserij', 'Macro-economie en belastingen', 'Wetenschappelijk onderzoek, technologie en communicatie', 'Toegevoegde codes voor media', 'Buitenlandse handel', 'Kunst, cultuur en entertainment', 'Energiebeleid', 'Ruimtelijke ordening, publiek natuur- en waterbeheer' ,'*** Sport', '*** Gemeentelijk en provinciaal bestuur', 'Ruimtelijke ordening, publiek natuur- en waterbeheer', 'Toegevoegde codes voor media']
    b = ['Overige' ] * len(a)
    overige_cat = dict(zip(a,b))

    df['main_topic_label'].replace(overige_cat, inplace = True)
    df['topic_label_dictionary'].replace(overige_cat, inplace = True)
    df['stemmed_topic_label_dictionary'].replace(overige_cat, inplace = True)

    logger.info("the length of categories identified by dict is now: {} ".format(len(df['topic_label_dictionary'].unique()) ) )
    logger.info("...and the stemmed dict: {} ".format(len(df['stemmed_topic_label_dictionary'].unique()) ) )
    return df

def apply_minnummatches():

    ''' specify how many words should match before the topic is considered present'''
    df = recode_dictionary()
    df['topic_label_dictionary_minmatches'] = np.where(df['len matches'] < MINNUMBERMATCHES, 'Overige', df['topic_label_dictionary'])
    df['topic_label_dictionary_minmatches_stem'] = np.where(df['stemmed_len matches'] < MINNUMBERMATCHES, 'Overige', df['stemmed_topic_label_dictionary'])

    ref_cols = ['text_x', 'main_topic_label', 'hmnintrst_wrds','attrresp', 'attrresp_wrds', 'cnflct', 'cnflct_wrds', 'ecnmc', 'ecnmc_wrds','hmnintrst', 'hmnintrst_wrds']
    df = df.loc[~df[ref_cols].duplicated()]
    df['main_topic_label'] = df['main_topic_label'].replace(np.nan, 'Overige')

    return df

def get_tp_fp_fn():

    '''create columns with true postives, false positives, and false negatives'''

    df = apply_minnummatches()
    topics = list(df['main_topic_label'].unique())

    for topic in topics:
        columnname_tp = "_tp " + str(topic)
        columnname_fp = "_fp " + str(topic)
        columnname_fn = "_fn " + str(topic)
        columnname_tn = "_tn " + str(topic)

        # and for stemmed

        columnname_tp_st = "st_tp " + str(topic)
        columnname_fp_st = "st_fp " + str(topic)
        columnname_fn_st = "st_fn " + str(topic)
        columnname_tn_st = "st_tn " + str(topic)

        # true positives = dictionary correctly identified.
        df[columnname_tp] = np.where( (df['main_topic_label'] == topic) & (df['topic_label_dictionary_minmatches'] == topic) , 1, 0 )
        # false positive = dictionary identified, but golden standard not.
        df[columnname_fp] = np.where( (df['main_topic_label'] != topic) & (df['topic_label_dictionary_minmatches'] == topic) , 1, 0 )
        # false negative = dictionary NOT identified, but golden standard DID identify
        df[columnname_fn] = np.where( (df['main_topic_label'] == topic) & (df['topic_label_dictionary_minmatches'] != topic) , 1, 0 )
        df[columnname_tn] = np.where( (df['main_topic_label'] != topic) & (df['topic_label_dictionary_minmatches'] != topic) , 1, 0 )

        # and for stemmed:
        df[columnname_tp_st] = np.where( (df['main_topic_label'] == topic) & (df['topic_label_dictionary_minmatches_stem'] == topic) , 1, 0 )
        df[columnname_fp_st] = np.where( (df['main_topic_label'] != topic) & (df['topic_label_dictionary_minmatches_stem'] == topic) , 1, 0 )
        df[columnname_fn_st] = np.where( (df['main_topic_label'] == topic) & (df['topic_label_dictionary_minmatches_stem'] != topic) , 1, 0 )
        df[columnname_tn_st] = np.where( (df['main_topic_label'] != topic) & (df['topic_label_dictionary_minmatches_stem'] != topic) , 1, 0 )

    return df

def main():
    df = get_tp_fp_fn()
    print("Done! Created a df with {} cases.........".format(len(df)))

    print("\n\nA sample to check whether all went okay: ")
#    print(df[df['main_topic_label'] == 'Overige'][['main_topic_label', 'topic_label_dictionary_minmatches', '_tp Overige', '_fp Overige', '_fn Overige']].head(10))
    df.to_pickle('{}RPA_data_with_dictionaryscores.pkl'.format(PATH_TO_DATA))

if __name__ == '__main__':
    logger = logging.getLogger()
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)

    main()
