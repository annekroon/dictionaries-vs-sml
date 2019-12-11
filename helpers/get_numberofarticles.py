#!/usr/bin/env python3
import json
import os
import glob
import pandas as pd

path_to_jsonfiles = '../output/uniekezinnen_2000-01-01_2018-12-31_numberofarticles.json'


full_filename = "{}".format(path_to_jsonfiles)
with open(full_filename,'r') as fi:
    mydict = json.load(fi)

df = pd.DataFrame(mydict)
df.to_csv('../output/embedding_corpus.csv')
