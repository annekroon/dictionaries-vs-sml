import gensim
import logging
import re

lettersanddotsonly = re.compile(r'[^a-zA-Z\.]')

PATH = "../data/raw/embeddings/"
outputpath='../data/'
FILENAME = "political_news_corpus.txt"

#w2v_params = {
#    'size': 300,
#    'window': 10,
#    'negative': 15
#}

w2v_params = {
    'size': 150,
    'window': 10,
    'negative': 15
}
def preprocess(s):
    s = s.lower().replace('!','.').replace('?','.')  # replace ! and ? by . for splitting sentences
    s = lettersanddotsonly.sub(' ',s)
    return s

class train_model():

    def __init__(self):
        self.sentences = gensim.models.word2vec.PathLineSentences(PATH + FILENAME)

        self.model = gensim.models.Word2Vec(**w2v_params)
        self.model.build_vocab(self.sentences)
        print('Build Word2Vec vocabulary')
        self.model.train(self.sentences,total_examples=self.model.corpus_count, epochs=self.model.iter)
        print('Estimated Word2Vec model')

def train_and_save():
    filename = f"{outputpath}w2v_size_150_window_10_negative_15"
    casus = train_model()

    with open(filename, mode='wb') as fo:
        casus.model.save(fo)
    print('Saved model')
    print("reopen it with m = gensim.models.word2vec.load('{}')".format(filename))
    del(casus)

if __name__ == "__main__":

    logger = logging.getLogger()
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)

    train_and_save()
