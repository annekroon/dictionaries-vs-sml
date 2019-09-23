import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

pd.options.mode.chained_assignment = None

RAW_DATA = "/Users/anne/surfdrive/uva/projects/RPA_KeepingScore/data/RPA_coding.csv"

def clean(df):
    ''' returns a cleaned df'''

    logger.info("length df BEFORE removing articles without political content: {}".format(len(df)) )
    df = df[df['political_content'] != 2] # delete articles that are not political content
    logger.info("length df AFTER removing without political content: {}".format(len(df)) )
    df['doc_num'] = df['doc_number'].str.split('-').str[-1]     # get document number in the right format
    incorrect_entries = ["51U9209", "2000.=", "1815.pdf"]  # delete incorrect entries
    df = df[~df['doc_num'].isin(incorrect_entries)]
    return df

def recode_dateformat(df):
    ''' gets date in right format'''

    #df = clean()
    error = df.publication_date[pd.isnull(pd.to_datetime(df.publication_date, errors ='coerce'))]
    indexes_to_keep = set(range(df.shape[0])) - set(error.index)
    df = df.take(list(indexes_to_keep))
    df['publication_date'] = pd.to_datetime(df.publication_date, errors ='coerce')
    df['year'] = pd.DatetimeIndex(df['publication_date']).year
    df['month'] = pd.DatetimeIndex(df['publication_date']).month
    return df

def label_np(row):
    if row['doctype_or'] == 2:
        return 'volkskrant (print)'
    if row['doctype_or'] == 1:
        return 'telegraaf (print)'

def label_topic(row):
    if row['main_topic'] == '1':
        return 'Macro-economie en belastingen'
    if row['main_topic'] == '2':
        return 'Burgerrechten en vrijheden'
    if row['main_topic'] == '3':
        return 'Gezondheid'
    if row['main_topic'] == '4':
        return 'Landbouw en Visserij'
    if row['main_topic'] == '5':
        return 'Arbeid'
    if row['main_topic'] == '6':
        return 'Onderwijs'
    if row['main_topic'] == '7':
        return 'Milieu'
    if row['main_topic'] == '8':
        return 'Energiebeleid'
    if row['main_topic'] == '9':
        return 'Immigratie en integratie'
    if row['main_topic'] == '10':
        return 'Verkeer en vervoer'
    if row['main_topic'] == '11':
        return 'Unkown'
    if row['main_topic'] == '12':
        return 'Justitie, Rechtspraak, Criminaliteit'
    if row['main_topic'] == '13':
        return 'sociale Zaken'
    if row['main_topic'] == '14':
        return 'Gemeenschapsontwikkeling, huisvestingsbeleid en stedelijke planning'
    if row['main_topic'] == '15':
        return 'Ondernemingen, Bankwezen en binnenlandse handel '
    if row['main_topic'] == '16':
        return 'Defensie'
    if row['main_topic'] == '17':
        return 'Wetenschappelijk onderzoek, technologie en communicatie'
    if row['main_topic'] == '18':
        return 'Buitenlandse handel'
    if row['main_topic'] == '19':
        return 'Buitenlandse zaken en ontwikkelingssamenwerking'
    if row['main_topic'] == '20':
        return 'Functioneren democratie en openbaar bestuur'
    if row['main_topic'] == '21':
        return 'Ruimtelijke ordening, publiek natuur- en waterbeheer'
    if row['main_topic'] == '22':
        return 'Unkown 2'
    if row['main_topic'] == '23':
        return 'Kunst, cultuur en entertainment'
    if row['main_topic'] == '00':
        return 'Toegevoegde codes voor media'

def recode_maintopics(df):
    ''' returns recoded main topics '''

    df['main_topic'] = df['topic_number'].str[:-2]
    df['main_topic_label'] = df.apply (lambda row: label_topic (row),axis=1)
    df[['main_topic','main_topic_label']][0:5]
    return df

def create_dummies(df):
    ''' adds a list of dummy variables to the data frame, capturing whether a main topic is present or not '''
    s = pd.Series(list(df['main_topic'])).astype('category')
    dummies = pd.get_dummies(s)
    l = dummies.columns.to_list()
    column_names_all = ['mt_'+ str(e) for e in l ]
    dummies.columns = column_names_all
    column_names_keep = ['mt_'+ str(i) for i in range(1,11) ] + ["mt_00", "mt_23"] + ['mt_'+ str(i) for i in range(12,22)]  #['mt_22', 'mt_11']
    dummies = dummies[column_names_keep]
    df.reset_index(inplace = True) #reset index before merge
    df = df.join(dummies)
    return df

def read_and_clean():
    df = pd.read_csv(RAW_DATA, skiprows=1)
    delete = df.iloc[:,0:11]  + df.iloc[:,-4: ]
    df.drop(delete, axis=1, inplace = True)
    #column_names = list(df.columns.values)
    new_column_names = ['Codeursnaam', 'type_content', 'doctype_or', 'political_content', 'publication_date', 'doc_number', 'topic_number', 'words_topic', 'frames', 'attrresp', 'attrresp_wrds', 'frames', 'hmnintrst', 'hmnintrst_wrds', 'frames', 'cnflct', 'cnflct_wrds', 'frames', 'ecnmc', 'ecnmc_wrds']
    df.columns = new_column_names
    df['doctype'] = df.apply(lambda row: label_np (row),axis=1)

    print("reading and cleaning dataset ...")
    df = clean(df)
    df = recode_dateformat(df)
    df = recode_maintopics(df)
    df = create_dummies(df)
    df = df[:-1] # last row = empty
    return df
