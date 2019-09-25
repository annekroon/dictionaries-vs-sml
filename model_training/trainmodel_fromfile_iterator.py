import gensim
import logging
import re

lettersanddotsonly = re.compile(r'[^a-zA-Z\.]')

PATH = "/home/anne/tmpanne/RPA/"
FILENAME = "parliamentaryquestions_news_corpus.txt"

w2v_params = {
    'size': 300,
    'window': 10,
    'negative': 15
}

def preprocess(s):
    s = s.lower().replace('!','.').replace('?','.')  # replace ! and ? by . for splitting sentences
    s = lettersanddotsonly.sub(' ',s)
    return s

class train_model():

    def __init__(self, fromdate,todate):
        self.fromdate = fromdate
        self.todate = todate

        self.sentences = gensim.models.word2vec.PathLineSentences(PATH + FILENAME)

        self.model = gensim.models.Word2Vec(**w2v_params)
        self.model.build_vocab(self.sentences)
        print('Build Word2Vec vocabulary')
        self.model.train(self.sentences,total_examples=self.model.corpus_count, epochs=self.model.iter)
        print('Estimated Word2Vec model')

def train_and_save(fromdate,todate):
    filename = "{}w2v_300d{}_{}".format(PATH,fromdate,todate)

    casus = train_model(fromdate,todate)

    with open(filename, mode='wb') as fo:
        casus.model.save(fo)
    print('Saved model')
    print("reopen it with m = gensim.models.word2vec.load('{}')".format(filename))
    del(casus)

if __name__ == "__main__":

    logger = logging.getLogger()
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)

    train_and_save(fromdate = "2000-01-01", todate = "2018-12-31")
