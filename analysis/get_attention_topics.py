
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import json

PATH = '/Users/anne/surfdrive/uva/projects/RPA_KeepingScore/data/RPA_data_with_dictionaryscores.pkl'

class get_figure():
    '''get figure'''

    def __init__(self, path_to_data):
        self.path_to_data = path_to_data
        self.path_to_output = "../figures/"
        with open('../resources/topic_translation') as handle:
               self.translator = json.loads(handle.read())

    def get_and_save_figure(self):
        df = pd.read_pickle(PATH)
        df['Policy topic'] = df['main_topic_label'].replace(self.translator)
        df['Data source'] = df['type'].replace({"parlementary question": "Parliamentary questions", "newspaper" : "Newspaper articles" })

        data = (df.groupby(['Data source'])['Policy topic']
                     .value_counts(normalize=True)
                     .rename('Percentage')
                     .mul(100)
                     .reset_index()
                     .sort_values('Percentage'))
        fig, ax = plt.subplots(figsize=(10,3))
        p = sns.barplot(x="Policy topic", y="Percentage", hue="Data source", data=data, palette=['gray', 'black'])
        plt.setp(p.get_xticklabels(), rotation=90)
        p = sns.set_style("white")
        fs = 16
        plt.ylabel('Relative attention (/%)', fontsize=fs)
        plt.xlabel(None)
        fname = '{}attention_topics'.format(self.path_to_output)
        fig.savefig(fname, bbox_inches='tight')
        plt.legend(prop={'size': 16})
        print('Saved figure as: {}'.format(fname))

fig = get_figure(PATH)
fig.get_and_save_figure()
