
import pandas as pd
import re
import logging
import os

PATH_TO_DATA = '/Users/anne/surfdrive/uva/projects/RPA_KeepingScore/data/'
FILE = 'parliamentary_questions_parsed.pkl'

def parse_parliamentary_questions():
    df = pd.read_pickle(PATH_TO_DATA + FILE)
    df['text'] = df['questions'].astype(str) + ' ' + df['answers'].astype(str)
    text = df['text'].to_list()
    sentence = [ re.split("(?<=[.!?])\s+", element) for element in text ]
    sentences = [item for sublist in sentence for item in sublist]
    logging.info("The total length of sentences is: {}...".format(len(sentences)))
    sentences = set(sentences)
    logging.info("...After removing duplicates, the length of the sentences is: {}".format(len(sentences)) )
    return sentences

def get_sentences():

    sentences = parse_parliamentary_questions()

    filename = '{}sentences_parliamentary_questions.txt'.format(PATH_TO_DATA)
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    with open(filename, mode='w') as fo:
        for s in sentences:
            fo.write(s)
            fo.write('\n')

    print("Created file with {} sentences: {}".format(len(sentences), filename))

if __name__ == "__main__":

    logger = logging.getLogger()
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)

    get_sentences()
