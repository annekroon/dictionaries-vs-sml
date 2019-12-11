from prep_annotated_data import *
import numpy as np
import pandas as pd
import logging

# merge annotated data with text from vk and tel

PATH_TO_DATA = '../data/raw/'

PATH_TO_VK = "/Volumes/AnneKroon/RPA/Media data/De Volkskrant/"
files_vk = os.listdir(PATH_TO_VK)

PATH_TO_TEL = "/Volumes/AnneKroon/RPA/Media data/De telegraaf/"
files_tel = os.listdir(PATH_TO_TEL)

print("lets go!")

df = read_and_clean()
print('get cleaned df')

files_xls_vk = [f for f in files_vk if f[-4:] == 'xlsx' and not f.startswith('.') and not f.startswith('~$') ]
files_xls_tel = [f for f in files_tel if f[-4:] == 'xlsx' and not f.startswith('.') and not f.startswith('~$') ]

df_vk = pd.DataFrame()
for f in files_xls_vk:
   # print(f)
    data = pd.read_excel(PATH_TO_VK + f)
    df_vk = df_vk.append(data)

df_tel = pd.DataFrame()
for f in files_xls_tel:
    print(f)
    data = pd.read_excel(PATH_TO_TEL + f)
    df_tel = df_tel.append(data)

df_vk['doc_num'] = df_vk['document nummer'].fillna(0) + df_vk['document numer'].fillna(0) + df_vk['Document nummer'].fillna(0)
df_vk['doc_num'] = pd.to_numeric(df_vk['doc_num'])
df_vk['publication_date'] = df_vk['publication_date'].str[:10]
df_vk['publication_date'] = pd.to_datetime(df_vk.publication_date, format = '%Y-%m-%d', errors="raise")

df_tel['doc_num'] = df_tel['document nummer'].fillna(0) + df_tel['Document nummer'].fillna(0)
df_tel['publication_date'] = df_tel['publication_date'].str[:10]
df_tel['publication_date'] = pd.to_datetime(df_tel.publication_date, format = '%Y-%m-%d', errors="raise")

df_all = pd.concat([df_vk, df_tel])

r = pd.merge(df, df_all, on=['doc_num' , 'doctype', 'publication_date'])
r = r[r['text'].notnull()]

mask = r.text.apply(lambda x: isinstance(x, bool))
mask2 = r.main_topic_label.apply(lambda x: isinstance(x, (bytes, type(None))))
#df = df[~mask]
r = r[~mask]
r = r[~mask2]


fname = '{}VK_TEL_merged_with_annotated'.format(PATH_TO_DATA)
r.to_pickle(fname)
