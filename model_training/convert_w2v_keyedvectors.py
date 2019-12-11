import gensim
from gensim.models import KeyedVectors
import logging
import os

PATH = '/home/anne/tmpanne/RPA/w2v_models/'

class word2vec_transformer():
    '''This class transforms word2vec models to keyedvectors'''

    def __init__(self, path_to_embeddings):
        self.nmodel = 0
        self.basepath = path_to_embeddings

    def get_w2v_model(self):
        '''yields a dict with one item. key is the filename, value the gensim model'''

        filenames = [e for e in os.listdir(self.basepath) if not e.startswith('.')]

        for fname in filenames:
            model = {}
            path = os.path.join(self.basepath, fname)
            print("\nLoading gensim model")

            if fname.startswith('w2v'):
                model['gensimmodel'] = KeyedVectors.load(path)
                model['filename'] = fname
                self.nmodel +=1
                print("loaded gensim model nr {}, named: {}".format(self.nmodel, model['filename']))

                yield model

    def save_model(self):
        for model in self.get_w2v_model():
            flnm = "{}{}.txt".format(self.basepath, model['filename'])
            logging.INFO("the new filename is: {}".format(flnm))
            model['gensimmodel'].wv.save_word2vec_format("{}".format(flnm))
            logging.INFO('Saved model')

            logging.INFO("reopen it with gensim.models.KeyedVectors.load_word2vec_format ('{}')".format(flnm) )

if __name__ == "__main__":

    logger = logging.getLogger()
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)

    transformer = word2vec_transformer(path_to_embeddings = PATH)
    transformer.save_model()
