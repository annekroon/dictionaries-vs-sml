import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


OUTPUTPATH = '../output/'
OUTPUTFIGURES = '../figures/'

translation = {'Functioneren democratie en openbaar bestuur' : 'Governmental operations' ,
 'Onderwijs' : 'Education' ,
 'Burgerrechten en vrijheden'  : 'Civil rights' ,
 'Justitie, Rechtspraak, Criminaliteit' : 'Law & crime' ,
 'Ondernemingen, Bankwezen en binnenlandse handel ' : 'Banking, finance, & commerce' ,
 'Defensie' : 'Defense' ,
 'Gezondheid' : 'Health' ,
 'Gemeenschapsontwikkeling, huisvestingsbeleid en stedelijke planning' : 'Community dev. & housing' ,
 'Verkeer en vervoer' : 'Transportation' ,
 'Buitenlandse zaken en ontwikkelingssamenwerking' : 'Int. affairs & foreign aid' ,
 'Macro-economie en belastingen' : 'Macroeconomics' ,
 'Wetenschappelijk onderzoek, technologie en communicatie': 'Science, technology & comm.' ,
 'Arbeid' : 'Labor & employment' ,
 'Overige' : 'Other issue' ,
 'Immigratie en integratie' : 'Immigration & integration',
 'sociale Zaken' : 'Social welfare',
 'Landbouw en Visserij' : 'Agriculture' ,
 'Energiebeleid' : 'Energy' ,
 'Milieu' : 'Environment'}


## Dictionary

def get_confusion_matrix(approach, sample, classifier = None):
    if approach == "Dictionary Approach":
        df = pd.read_pickle('/Users/anne/surfdrive/uva/projects/RPA_KeepingScore/data/RPA_and_Buschers_data_with_dictionaryscores.pkl')

        if sample == 'totalsample':
            df = df
        elif sample == 'newspaper_sample_only':
            df = df[df['type'] == 'newspaper']
        elif sample == 'pq_sample_only' :
            df = df[df['type'] == 'parlementary question']
        elif sample == 'RPA_sample' :
            df = df[df['origin'] == 'RPA']
        elif sample == 'Bjorns_sample' :
            df = df[df['origin'] == 'Bjorn']

        df = df[['main_topic_label', 'topic_label_dictionary']]
        df.rename(columns={'main_topic_label':'Actual label','topic_label_dictionary':'Predicted label'}, inplace=True)

    elif approach == 'SML':
        base = "{}SML_predicted_actual_{}.json".format(OUTPUTPATH, sample)
        print(base)
        df = pd.read_json(base)
        if classifier == "Passive Agressive":
            df = df[df['Classifier'] == "Passivie Aggressive"]
        if classifier == "SGDClassifier":
            df = df[df['Classifier'] == "SGDClassifier"]
        if classifier == "Naive Bayes":
            df = df[df['Classifier'] == "Naive Bayes"]

    df.replace(translation, inplace=True)
    confusion_matrix = pd.crosstab(df['Actual label'], df['Predicted label'], rownames=['True'], colnames=['Predicted'])
    cmn = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
    return cmn

def get_heatmap(approach, sample, classifier):
    label = "Values in the diagonal represent the relative times that the manual coding (’true label’ - Y axis ) is equal to the classifier (X axis). Diagonal values indicate the relative number of correct predictions: The higher the values in diagonal, the better the prediction. Off-diagonal values indicate misclassification. Darker colours indicate higher values. Due to class imbalance, values are normalised to facilitate visual understanding. Values below 0.1 are not visualised. "
    cmn = get_confusion_matrix(approach, sample, classifier)
    cmn = cmn.round(1)
    fig, ax = plt.subplots(figsize=(10,10))
    heatmap = sns.heatmap(cmn, annot=True, annot_kws={"size": 10}, fmt='.1f',  cmap="YlGnBu", mask=(cmn<0.1))
    fs = 12
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fs)
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fs)
    plt.title("Normalized Confusion Matrix (Classifier: {}) ".format(approach), fontsize= 16)
    plt.ylabel('True label (Manual coding)', fontsize=fs)
    plt.xlabel('Predicted label', fontsize=fs)
    return fig

def get_figure_save(approach, sample, classifier = None):
    #logger.info('{}'.format(label))
    figure = get_heatmap(approach, sample, classifier = None)
    if approach == 'SML':
        fname = '{}Heatmap_{}_{}_{}'.format(OUTPUTFIGURES, approach, classifier, sample)
    else:
         fname = '{}Heatmap_{}_{}'.format(OUTPUTFIGURES, approach, sample)
    figure.savefig(fname, bbox_inches='tight')
    print('Saved figure as: {}'.format(fname))

get_figure_save('Dictionary Approach', 'totalsample')
get_figure_save('Dictionary Approach', 'RPA_sample')
get_figure_save('Dictionary Approach', 'Bjorns_sample')
get_figure_save('Dictionary Approach', 'newspaper_sample_only')
get_figure_save('Dictionary Approach', 'parlementary question')

get_figure_save('SML', 'totalsample', 'Passive Aggressive')
get_figure_save('SML', 'totalsample', 'Naive Bayes')
get_figure_save('SML', 'totalsample', 'SGDClassifier')

get_figure_save('SML', 'RPA_sample', 'Passive Aggressive')
get_figure_save('SML', 'RPA_sample', 'Naive Bayes')
get_figure_save('SML', 'RPA_sample', 'SGDClassifier')
