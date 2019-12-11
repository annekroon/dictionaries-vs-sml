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
 'macro avg' : 'Accuracy'}



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
        df2['classifier'] = 'Albaugh et al. - not stemmed'

        df = pd.concat([df, df2])
        df.rename(columns={0 :'precision',  1 :'recall', 2 :'f1-score'}, inplace=True)
        df.rename(index=self.translator, inplace=True)
        df['approach'] = 'Dictionary Approach'
        return df


    def get_data_sml(self, vect):
        fname_sml = '{}sml_vectorizers_final/SML_precision_recall_f1score_text_cleaned_{}_{}.json'.format(self.path_to_data, self.sample, vect)
        fname_sml= fname_sml
        with open(fname_sml) as handle:
            class_report =  json.loads(handle.read())

        one, two, three, four = class_report
        print(four)
        one = pd.DataFrame(one).transpose()
        one['classifier'] = "Passive Agressive"
        logging.info("Results Gridsearch Passive Agressive: \n\nC: {} \nfit intercept: {} \nmax_iter: {}  \nloss: {}"
              .format(one['clf__C'][one.index=='best estimators:'] [0] ,
                      one['clf__fit_intercept'][one.index=='best estimators:'][0] ,
                      one['clf__max_iter'][one.index=='best estimators:'] [0] ,
                      one['clf__loss'][one.index=='best estimators:'][0]  )  )


        two = pd.DataFrame(two).transpose()
        two['classifier'] = "Stochastic Gradient Descent (SGD)"

        logging.info("Results Gridsearch SGD Classifier: \n\nAlpha: {} \nmax iter: {} \npenalty: {} \n\n."
              .format(two['clf__alpha'][two.index=='best estimators:'] [0] ,
                    two['clf__max_iter'][two.index=='best estimators:'][0] ,
                    two['clf__penalty'][two.index=='best estimators:'][0]) )

        three = pd.DataFrame(three).transpose()
        three['classifier'] = "Support Vector Machines (SVM)"

        logging.info("Results Gridsearch SVM Classifier: \n\nC: {} \ngamma: {} \nkernel: {} \n\n."
                      .format(three['clf__C'][three.index=='best estimators:'] [0] ,
                              three['clf__gamma'][three.index=='best estimators:'] [0] ,
                              three['clf__kernel'][three.index=='best estimators:'] [0] ))

        four = pd.DataFrame(four).transpose()
        four['classifier'] = "ExtraTrees"

        logging.info("Results Gridsearch ExtraTrees: \n\nMax features:{}".format(four['clf__max_features'][four.index=='best estimators:'] [0]  ))

        df_sml = pd.concat([one, two, three, four])
        df_sml = df_sml[['precision', 'recall', 'f1-score', 'classifier']]
        df_sml.drop(['best estimators:', 'classifier:'], inplace = True)
        #df_sml['approach'] = 'SML'
        df_sml.rename(index=self.translator, inplace=True)
        return df_sml

    def combine_datasets(self):
        df1 = self.get_data_dictionary()
        df2 = self.get_data_sml(vect='w2v_tfidf')
        df2['approach'] = 'w2v tfidf'
        df3 = self.get_data_sml(vect='tfidf')
        df3['approach'] = 'tfidf'
        df4 = self.get_data_sml(vect='w2v_count')
        df4['approach'] = 'w2v count'
        df5 = self.get_data_sml(vect='count')
        df5['approach'] = 'count'
        df = pd.concat([df1, df2, df3, df4, df5])
        df['Policy topic'] = df.index
        df.rename(index={'macro avg': 'Accuracy'}, inplace=True)
        df.replace({'macro avg': 'Accuracy'}, inplace=True)
       # df.drop(['macro avg', 'Average'], inplace = True)
        return df


def get_figure_and_save():
    myanalyzer = plot_accuracy_precision_recall(path_to_data = '../output/', path_to_output = '../tables/', sample ='newspaper_sample_only')
    df = myanalyzer.combine_datasets()
    fname = '{}classification_topics'.format('../figures/')

    df['classifier + vectorizer'] = df['classifier'].astype(str) + " ~ " + df['approach'].astype(str)
    accuracy = df[df['Policy topic'] == 'Accuracy']
    approach = accuracy["approach"]
    colour = ['whitesmoke' if x=='w2v tfidf' else 'dimgray' if x== 'w2v count' else 'black' if x== 'count' else 'silver' if x== 'tfidf' else 'white' for x in approach ]

    final_recode = {'Albaugh et al. - not stemmed ~ Dictionary Approach' : 'Albaugh et al. - not stemmed (dictionary)' ,
     'Albaugh et al. (Dictionary) - stemmed ~ Dictionary Approach' : 'Albaugh et al. - stemmed (dictionary)' ,
     'Support Vector Machines (SVM) ~ w2v count': 'SVM count embedding' ,
     'Support Vector Machines (SVM) ~ w2v tfidf' : 'SVM tfidf embedding' ,
     'ExtraTrees ~ count' : 'ET count',
     'ExtraTrees ~ tfidf' : 'ET tfidf',
     'ExtraTrees ~ w2v count' : 'ET count embedding',
     'Support Vector Machines (SVM) ~ tfidf' : 'SVM tfidf',
     'Support Vector Machines (SVM) ~ count' :  'SVM count',
     'ExtraTrees ~ w2v tfidf' :  'ET tfidf embedding',
     'Passive Agressive ~ tfidf' :  'PA tfidf',
     'Passive Agressive ~ count' :  'PA count',
     'Passive Agressive ~ w2v count' :  'PA count embedding',
     'Passive Agressive ~ w2v tfidf' :  'PA tfidf embedding',
     'Stochastic Gradient Descent (SGD) ~ count' :  'SGD count',
     'Stochastic Gradient Descent (SGD) ~ w2v count' : 'SGD count embedding',
     'Stochastic Gradient Descent (SGD) ~ tfidf' : 'SGD tfidf',
     'Stochastic Gradient Descent (SGD) ~ w2v tfidf' : 'SGD tfidf embedding'}

    df['classifier_updated'] = df['classifier + vectorizer'].map(final_recode)
    print(df['classifier_updated'] )
    f, ax = plt.subplots(figsize=(6,10))
    sns.set_context('talk')
    sns.set(style="whitegrid")

    order = ['Albaugh et al. - stemmed (dictionary)', 'Albaugh et al. - not stemmed (dictionary)',  'SVM tfidf', 'SVM tfidf embedding', 'SVM count', 'SVM count embedding', 'PA tfidf', 'PA tfidf embedding', 'PA count', 'PA count embedding', 'SGD tfidf', 'SGD tfidf embedding', 'SGD count', 'SGD count embedding', 'ET tfidf', 'ET tfidf embedding', 'ET count', 'ET count embedding']

    ax = sns.barplot(y="classifier_updated", x="f1-score",edgecolor=".4", palette=colour, order =order, data=df[df['Policy topic'] == 'Accuracy'])
    ax = sns.set_style("white")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.title(None)
    plt.ylabel(None)
    plt.xlabel(None)
    plt.savefig(fname, bbox_inches = 'tight')
    print('Saved figure as: {}'.format(fname))

    sorterIndex = dict(zip(order,range(len(order))))
    # Generate a rank column that will be used to sort
    # the dataframe numerically
    df['Tm_Rank'] = df['classifier_updated'].map(sorterIndex)
    df.sort_values(['Policy topic','Tm_Rank'], inplace=True)

    df = df[df['Policy topic'].isin(['Accuracy'])]
    df.to_csv('../output/accuracy_topics.csv')

get_figure_and_save()
