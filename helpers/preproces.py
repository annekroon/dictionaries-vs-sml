from nltk.corpus import stopwords
from string import punctuation
import nltk
import pandas as pd

PATH = '/Users/anne/surfdrive/uva/projects/RPA_KeepingScore/data/RPA_data_with_dictionaryscores.pkl'
df = pd.read_pickle(PATH)
body_of_text = df['text_x'].to_list()

print(body_of_text[0][:501])

body_of_text=["".join([l for l in speech if l not in punctuation]) for speech in body_of_text]  #remove punctuation
body_of_text=[speech.lower() for speech in body_of_text]  # convert to lower case
body_of_text=[" ".join(speech.split()) for speech in body_of_text]

mystopwords = stopwords.words('dutch')

extra_stop = [line.strip() for line in open('../stopwords/stopwords_NL.txt').readlines() if len(line)>1]
mystopwords = set(mystopwords + extra_stop)

text_clean = [" ".join([w for w in speech.split() if w not in mystopwords]) for speech in body_of_text]
text_clean = [" ".join([w for w in speech.split() if w.isalpha()]) for speech in text_clean] # keep only words, no digits

print(text_clean[0][:501])

df['text_clean'] = text_clean

print("DONE! added column with cleaned text")
df.to_pickle(PATH)
print("SAVED")
