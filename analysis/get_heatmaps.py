import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


OUTPUTPATH = '../output/'
OUTPUTFIGURES = '../figures/'

df = pd.read_pickle('/Users/anne/surfdrive/uva/projects/RPA_KeepingScore/data/RPA_and_Buschers_data_with_dictionaryscores.pkl')
df['main_topic_id'] = df['main_topic_label'].factorize()[0]
num = df.index.to_list()
d = dict(zip(num, topics))

class get_heatmaps():
    '''Get those heatmaps'''

    def __init__(self, approach, sample, classifier = None):
        self.approach = approach
        self.sample = sample
        self.classifier = classifier
        with open('../resources/topic_translation') as handle:
               self.translator = json.loads(handle.read())

        with open('../resources/numbers_to_topic.json') as handle:
               self.translator_numbers = json.loads(handle.read())


    def get_data(self):

        if self.approach == "Dictionary Approach":
            df = pd.read_pickle('/Users/anne/surfdrive/uva/projects/RPA_KeepingScore/data/RPA_and_Buschers_data_with_dictionaryscores.pkl')

            if self.sample == 'totalsample':
                df = df
            elif self.sample == 'newspaper_sample_only':
                df = df[df['type'] == 'newspaper']
            elif self.sample == 'pq_sample_only' :
                df = df[df['type'] == 'parlementary question']
            elif self.sample == 'RPA_sample' :
                df = df[df['origin'] == 'RPA']
            elif self.sample == 'Bjorns_sample' :
                df = df[df['origin'] == 'Bjorn']

            df = df[['main_topic_label', 'topic_label_dictionary']]
            df.rename(columns={'main_topic_label':'Actual label','topic_label_dictionary':'Predicted label'}, inplace=True)
            return df

        elif self.approach == 'SML':
            base = "{}SML_predicted_actual_{}.json".format(OUTPUTPATH, sample)
            print(base)
            df = pd.read_json(base)
            if self.classifier == "Passive Agressive":
                df = df[df['Classifier'] == "Passivie Aggressive"]
            if self.classifier == "SGDClassifier":
                df = df[df['Classifier'] == "SGDClassifier"]
            if self.classifier == "Naive Bayes":
                df = df[df['Classifier'] == "Naive Bayes"]
            return df

        elif self.approach == 'CNN':
            cnn_file = "{}output/predicted_actual.csv".format(OUTPUTPATH)
            df = pd.read_csv(cnn_file, sep='\t', header=None, names = ['Predicted label', 'Actual label'])
            df.replace(d, inplace=True)
            return df

    def confusion_matrix(self):
        df = self.get_data()
        df.replace(self.translator, inplace=True)
        print(df.columns)
        confusion_matrix = pd.crosstab(df['Actual label'], df['Predicted label'], rownames=['True'], colnames=['Predicted'])
        cmn = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        return cmn

    def get_heatmap(self):
        label = "Values in the diagonal represent the relative times that the manual coding (’true label’ - Y axis ) is equal to the classifier (X axis). Diagonal values indicate the relative number of correct predictions: The higher the values in diagonal, the better the prediction. Off-diagonal values indicate misclassification. Darker colours indicate higher values. Due to class imbalance, values are normalised to facilitate visual understanding. Values below 0.1 are not visualised. "
        cmn = self.confusion_matrix()
        cmn = cmn.round(1)
        fig, ax = plt.subplots(figsize=(10,10))
        heatmap = sns.heatmap(cmn, annot=True, annot_kws={"size": 10}, fmt='.1f',  cmap="YlGnBu", mask=(cmn<0.1))
        fs = 12
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fs)
        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fs)
        plt.title("Normalized Confusion Matrix (Classifier: {}) ".format(self.approach), fontsize= 16)
        plt.ylabel('True label (Manual coding)', fontsize=fs)
        plt.xlabel('Predicted label', fontsize=fs)
        return fig

    def get_figure_save(self):
        #logger.info('{}'.format(label))
        figure = self.get_heatmap()
        if self.approach == 'SML':
            fname = '{}Heatmap_{}_{}_{}'.format(OUTPUTFIGURES, self.approach, self.classifier, self.sample)
        else:
             fname = '{}Heatmap_{}_{}'.format(OUTPUTFIGURES, self.approach, self.classifier, self.sample)
        figure.savefig(fname, bbox_inches='tight')
        print('Saved figure as: {}'.format(fname))

a = get_heatmaps(approach = 'CNN', sample = 'totalsample')
a.get_figure_save()

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
