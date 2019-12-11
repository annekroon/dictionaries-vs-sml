import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json



OUTPUTPATH = '../output/'
OUTPUTFIGURES = '../figures/'

df = pd.read_pickle('/Users/anne/surfdrive/uva/projects/RPA_KeepingScore/data/RPA_data_with_dictionaryscores.pkl')
df['main_topic_id'] = df['main_topic_label'].factorize()[0]
d = df.groupby('main_topic_id')['main_topic_label'].max().to_dict()

d2 = {"Banking, finance, & commerce":	1	,
"Civil rights":	2	,
"Defense":	3	,
"Education":	4	,
"Environment":	5	,
"Governmental operations":	6	,
"Health":	7	,
"Immigration & integration":	8	,
"Int. affairs & foreign aid":	9	,
"Labor & employment":	10	,
"Law & crime":	11	,
"Other issue":	12	,
"Social welfare":	13	,
"Transportation":	14	}

class get_heatmaps():
    '''Get those heatmaps'''

    def __init__(self, approach, sample, classifier = None, vect=None):
        self.approach = approach
        self.sample = sample
        self.classifier = classifier
        self.vect = vect
        with open('../resources/topic_translation') as handle:
               self.translator = json.loads(handle.read())

    def get_data(self):

        if self.approach == 'Dictionary Approach':
            df = pd.read_pickle('/Users/anne/surfdrive/uva/projects/RPA_KeepingScore/data/RPA_data_with_dictionaryscores.pkl')

            if self.sample == 'newspaper_sample_only':
                df = df[df['type'] == 'newspaper']
            elif self.sample == 'pq_sample_only' :
                df = df[df['type'] == 'parlementary question']
            elif self.sample == 'RPA_sample' :
                df = df[df['origin'] == 'RPA']


            print("Dataframe with the sample: {}".format(self.sample))
            print("The length of the dataframe is: {}".format(len(df)))
            df = df[['main_topic_label', 'topic_label_dictionary']]
            df.rename(columns={'main_topic_label':'Actual label','topic_label_dictionary':'Predicted label'}, inplace=True)
            return df

        elif self.approach == 'SML':
            base = "{}sml_vectorizers_final/SML_predicted_actual_text_cleaned_{}_{}.json".format(OUTPUTPATH, self.sample, self.vect)
            print(base)
            df = pd.read_json(base)
            if self.classifier == "Passive Agressive":
                df = df[df['Classifier'] == "Passivie Aggressive"]
            if self.classifier == "SGDClassifier":
                df = df[df['Classifier'] == "SGDClassifier"]
            if self.classifier == "Naive Bayes":
                df = df[df['Classifier'] == "Naive Bayes"]
            return df

    def confusion_matrix(self):
        df = self.get_data()
        df.replace(self.translator, inplace=True)
        df.replace(d2, inplace=True)
        confusion_matrix = pd.crosstab(df['Actual label'], df['Predicted label'], rownames=['True'], colnames=['Predicted'])
        cmn = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        return cmn

    def get_heatmap(self):
        label = "Values in the diagonal represent the relative times that the manual coding (’true label’ - Y axis ) is equal to the classifier (X axis). Diagonal values indicate the relative number of correct predictions: The higher the values in diagonal, the better the prediction. Off-diagonal values indicate misclassification. Darker colours indicate higher values. Due to class imbalance, values are normalised to facilitate visual understanding. Values below 0.1 are not visualised. "
        cmn = self.confusion_matrix()
        cmn = cmn.round(1)
        fig, ax = plt.subplots(figsize=(10,10))
    #    heatmap = sns.heatmap(cmn, annot=True, annot_kws={"size": 10}, fmt='.1f',  cmap="BuGn", mask=(cmn<0.1))
        heatmap = sns.heatmap(cmn, annot=True, annot_kws={"size": 10}, fmt='.1f',  cmap="gist_gray_r", linecolor='black', mask=(cmn<0.1))
        for _, spine in heatmap.spines.items():
            spine.set_visible(True)
        fs = 16
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fs)
        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fs)
        #plt.title("Normalized Confusion Matrix (Classifier: {}) ".format(self.approach), fontsize= 16)
        plt.title(None)
        plt.ylabel('True label (Manual coding)', fontsize=fs)
        plt.xlabel('Predicted label', fontsize=fs)
        return fig

    def get_figure_save(self):
        #logger.info('{}'.format(label))
        figure = self.get_heatmap()
        if self.approach == 'SML':
            fname = '{}Heatmap_{}_{}_{}'.format(OUTPUTFIGURES, self.approach, self.classifier, self.sample)
        else:
             fname = '{}Heatmap_{}_{}'.format(OUTPUTFIGURES, self.approach, self.sample)
        figure.savefig(fname, bbox_inches='tight')
        print('Saved figure as: {}'.format(fname))

a = get_heatmaps(approach = 'SML', classifier='SGDClassifier', sample = 'RPA_sample', vect='w2v_count')
a.get_figure_save()

a = get_heatmaps(approach = 'Dictionary Approach',  sample = 'RPA_sample')
a.get_figure_save()
