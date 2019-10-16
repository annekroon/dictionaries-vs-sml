import seaborn as sns
import matplotlib.pyplot as plt
import json
import logging
import pandas as pd

d = {'0': 'Onderwijs',
 '1': 'Burgerrechten en vrijheden',
 '2': 'Justitie, Rechtspraak, Criminaliteit',
 '3': 'Defensie',
 '4': 'Gezondheid',
 '5': 'Gemeenschapsontwikkeling, huisvestingsbeleid en stedelijke planning',
 '6': 'Functioneren democratie en openbaar bestuur',
 '7': 'Macro-economie en belastingen',
 '8': 'Buitenlandse zaken en ontwikkelingssamenwerking',
 '9': 'Ondernemingen, Bankwezen en binnenlandse handel ',
 '10': 'Arbeid',
 '11': 'Verkeer en vervoer',
 '12': 'Overige',
 '13': 'sociale Zaken',
 '14': 'Immigratie en integratie',
 '15': 'Landbouw en Visserij',
 '16': 'Energiebeleid',
 '17': 'Milieu',
 '18': 'Wetenschappelijk onderzoek, technologie en communicatie',
 'micro avg' : 'Accuracy'}



logger = logging.getLogger()
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
logging.root.setLevel(level=logging.INFO)

class plot_accuracy_precision_recall():
    '''This prepares a CNN model and runs it'''

    def __init__(self, path_to_data, path_to_output, sample):
        self.path_to_data = path_to_data
        self.path_to_output = path_to_output
        self.sample = sample
        with open('../resources/topic_translation') as handle:
               self.translator = json.loads(handle.read())
  #      with open('../resources/numbers_to_topic.json') as handle:
   #            self.translator_numeric = json.loads(handle.read())

    def get_data_dictionary(self):
         # getting Dictionary Approach Data
        fname = '{}precision_recall_f1score_dictionary_stemmed{}.json'.format(self.path_to_data, self.sample)
        logger.info(fname)

        with open(fname) as handle:
            dictdump =  json.loads(handle.read())

        df = pd.DataFrame.from_dict(dictdump).transpose()
        df['classifier'] = 'Albaugh et al. (Dictionary) - stemmed'

        fname_notstemmed = '{}precision_recall_f1score_dictionary_not_stemmed{}.json'.format(self.path_to_data, self.sample)
        logger.info(fname_notstemmed)

        with open(fname_notstemmed) as handle:
            dictdump =  json.loads(handle.read())

        df2 = pd.DataFrame.from_dict(dictdump).transpose()
        df2['classifier'] = 'Albaugh et al. (Dictionary) - not stemmed'

        df = pd.concat([df, df2])
        df.rename(columns={0 :'precision',  1 :'recall', 2 :'f1-score'}, inplace=True)
        df.rename(index=self.translator, inplace=True)
        df['approach'] = 'Dictionary Approach'
        return df


    def get_data_cnn(self):

        '''create if/else for different samples'''

        fname_cnn = '{}precision_recall_f1score_embedding_vectorizer_text_clean{}.json'.format(self.path_to_data, self.sample)
        with open(fname_cnn) as handle:
            data =  json.loads(handle.read())

        df = pd.DataFrame.from_dict(data).transpose()
        df['approach'] = 'Embedding Vectorizer'
 #       df.rename(index=self.translator, inplace=True)
        df['classifier'] = 'Random Trees + TfidF embedding Vectorizer'
       # df.index = df.index.map(int)
        df.rename(index=d, inplace=True)
        df.rename(index=self.translator, inplace=True)
   #     df = df[['f1-score', 'approach', 'classifier']]
        return df


    def get_data_sml(self):
        fname_sml = '{}SML_precision_recall_f1score_text_cleaned_{}.json'.format(self.path_to_data, self.sample)
        with open(fname_sml) as handle:
            class_report =  json.loads(handle.read())

        one, two, three, four = class_report
        one = pd.DataFrame(one).transpose()
        one['classifier'] = "Naive Bayes"
        logging.info("Results Gridsearch Naive Bayes: \n\nAlpha: {} \nNgram_range: {} \n\n."
              .format(one['clf__alpha'][one.index=='best estimators:'] [0] ,
                      one['vect__ngram_range'][one.index=='best estimators:'][0]) )

        two = pd.DataFrame(two).transpose()
        two['classifier'] = "Passive Agressive"

        logging.info("Results Gridsearch Passive Agressive: \n\nC: {} \nfit intercept: {} \nmax_iter: {} \nmax_loss: {} \nuse idf: {} \nvec ngram range: {}"
              .format(two['clf__C'][two.index=='best estimators:'] [0] ,
                      two['clf__fit_intercept'][two.index=='best estimators:'][0] ,
                      two['clf__max_iter'][two.index=='best estimators:'] [0] ,
                      two['clf__loss'][two.index=='best estimators:'][0] ,
                      two['tfidf__use_idf'][two.index=='best estimators:'] [0] ,
                      two['vect__ngram_range'][two.index=='best estimators:'][0] )  )

        three = pd.DataFrame(three).transpose()
        three['classifier'] = "Stochastic Gradient Descent (SGD)"

        logging.info("Results Gridsearch SGD Classifier: \n\nAlpha: {} \nmax iter: {} \npenalty: {} \n\n."
              .format(three['clf__alpha'][one.index=='best estimators:'] [0] ,
                      three['clf__max_iter'][one.index=='best estimators:'][0] ,
                      three['clf__penalty'][one.index=='best estimators:'][0]) )

        four = pd.DataFrame(four).transpose()
        four['classifier'] = "Support Vector Machines (SVM)"

        df_sml = pd.concat([one, two, three, four])
        df_sml = df_sml[['precision', 'recall', 'f1-score', 'classifier']]
        df_sml.drop(['best estimators:', 'classifier:'], inplace = True)
        df_sml['approach'] = 'SML'
        df_sml.rename(index=self.translator, inplace=True)
        return df_sml

    def combine_datasets(self):
        df1 = self.get_data_dictionary()
        df2 = self.get_data_sml()
        df3 = self.get_data_cnn()
        df = pd.concat([df1, df2, df3])
        df['Policy topic'] = df.index
        df.rename(index={'micro avg': 'Accuracy'}, inplace=True)
        df.replace({'micro avg': 'Accuracy'}, inplace=True)
       # df.drop(['macro avg', 'Average'], inplace = True)
        return df


def get_figure_and_save():
    myanalyzer = plot_accuracy_precision_recall(path_to_data = '../output/', path_to_output = '../tables/', sample ='RPA_sample')
    df = myanalyzer.combine_datasets()
    fname = '{}classification_topics'.format('../figures/')
#    fig.savefig(fname, bbox_inches='tight')
#    plt.legend(prop={'size': 16})

    f, ax = plt.subplots(figsize=(12,14))
    sns.set_context('talk')
    sns.set(style="whitegrid")

    order = ['Banking, finance, & commerce', 'Civil rights',
           'Defense', 'Education', 'Environment', 'Governmental operations',
           'Health', 'Immigration & integration',
           'Int. affairs & foreign aid', 'Labor & employment', 'Law & crime',
           'Social welfare', 'Transportation','Other issue', 'Accuracy']

    hue_order=['Albaugh et al. (Dictionary) - not stemmed', 'Albaugh et al. (Dictionary) - stemmed' ,'Naive Bayes',
           'Passive Agressive', 'Stochastic Gradient Descent (SGD)', 'Support Vector Machines (SVM)', 'Random Trees + Embedding Vectorizer' ]

    ax = sns.barplot(y="Policy topic", x="f1-score", hue = "classifier", edgecolor=".7", order = order, hue_order = hue_order , palette="ch:.25", data=df)
    ax = sns.set_style("white")
    plt.title(None)
    plt.ylabel(None)
    plt.xlabel(None)
    plt.savefig(fname, bbox_inches = 'tight')
    print('Saved figure as: {}'.format(fname))

get_figure_and_save()
