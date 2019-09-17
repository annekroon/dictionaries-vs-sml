from prep_annotated_data import *
import numpy as np
import pandas as pd
import logging

print("lets go!")

PATH_TO_DATA = '/Users/anne/surfdrive/uva/projects/RPA_KeepingScore/pickle_files/'
annotated = read_and_clean()
print('get cleaned df')

annotated = annotated[annotated['type_content'] == 2]
annotated['year'] = pd.DatetimeIndex(annotated['publication_date']).year


parsed_kv = pd.read_pickle('~/surfdrive/uva/projects/RPA_KeepingScore/data/parliamentary_questions_parsed.pkl')

def parse_identifier(row):
    try:
        id_number = row.split('-')[-1]
    except:
        id_number = np.nan
    return id_number

def parse_identifier_d(row):
    return row.replace('.', '-').split('-')[-3]

annotated['id_number'] = annotated['doc_number'].apply(lambda row: parse_identifier(row))


def split_date_year(row):
    try:
        date = row.split()[-1]
        date = pd.to_datetime(date, format='%Y')
        return date
    except:
        return np.nan

parsed_kv['year'] = parsed_kv['date_send_in'].apply(lambda row: split_date_year(row))
parsed_kv['year'] =  pd.DatetimeIndex(parsed_kv['year']).year

def parse_identifier(row):
    return row.replace('.xml', '').split('-')[-1]

parsed_kv['id_number'] = parsed_kv['filename'].apply(lambda row: parse_identifier(row))

parsed_kv['id_number'] = parsed_kv['id_number'].apply(lambda x: int(x) if str(x).isdigit() else None)
annotated['id_number'] = annotated['id_number'].apply(lambda x: int(x) if str(x).isdigit() else None)

parsed_kv['id_number'] = parsed_kv['id_number'].astype(str)
annotated['id_number'] = annotated['id_number'].astype(str)

annotated['year'] = annotated['year'].fillna(0).astype(int)
annotated['year'] = annotated['year'].astype(str)
parsed_kv['year'] = parsed_kv['year'].fillna(0).astype(int)
parsed_kv['year'] = parsed_kv['year'].astype(str)

df = pd.merge(annotated, parsed_kv, how= 'inner', on = ['year', 'id_number'])
print('merged annotated and parsed dataframe. The merged file has {} cases'.format(len(df)))

fname = '{}kamervragen'.format(PATH_TO_DATA)
df.to_pickle(fname)

print('saved df as {}'.format(fname))
