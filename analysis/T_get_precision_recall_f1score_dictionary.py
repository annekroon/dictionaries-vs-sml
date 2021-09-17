
import pandas as pd
import logging
import json
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score

PATH_TO_DATA = '../data/intermediate/'
FILENAME = 'RPA_data_with_dictionaryscores.pkl'

OUTPUT_PATH ='../output/'

def get_data():
    df = pd.read_pickle('../data/raw/RPA_data_with_dictionaryscores.pkl')

    df['topic_label_dictionary-freq'].replace(np.nan, 'Overige', inplace=True)
    df['stemmed_topic_label_dictionary-freq'].replace(np.nan, 'Overige', inplace=True)

    return df

def get_recall_precision(topics):

    true_positives = ["_tp " + str(i) for i in topics]
    false_positives = ["_fp " + str(i) for i in topics]
    false_negatives = ["_fn " + str(i) for i in topics]

    true_positives_freq = ["_tp_freq " + str(i) for i in topics]
    false_positives_freq = ["_fp_freq " + str(i) for i in topics]
    false_negatives_freq = ["_fn_freq " + str(i) for i in topics]

    true_positives_st = ["st_tp " + str(i) for i in topics]
    false_positives_st = ["st_fp " + str(i) for i in topics]
    false_negatives_st = ["st_fn " + str(i) for i in topics]

    true_positives_st_freq = ["st_tp_freq " + str(i) for i in topics]
    false_positives_st_freq = ["st_fp_freq " + str(i) for i in topics]
    false_negatives_st_freq = ["st_fn_freq " + str(i) for i in topics]


    recall = {}
    precision = {}
    f1score = {}
    accuracy = {}

    recall_freq = {}
    precision_freq = {}
    f1score_freq = {}
    accuracy_freq = {}

    recall_stemmed = {}
    precision_stemmed = {}
    f1score_stemmed = {}
    accuracy_stemmed = {}

    recall_stemmed_freq = {}
    precision_stemmed_freq = {}
    f1score_stemmed_freq = {}
    accuracy_stemmed_freq = {}

    df = get_data()


    for tp, fp, fn, st_tp, st_fp, st_fn, topic in zip(true_positives, false_positives, false_negatives, true_positives_st, false_positives_st, false_negatives_st, topics) :

        recall[str(topic)] = df[tp].sum(axis=0) / ( df[tp].sum(axis=0) + df[fn].sum(axis=0) )
        precision[str(topic)] = df[tp].sum(axis=0) / ( df[tp].sum(axis=0) + df[fp].sum(axis=0) )
        f1score[str(topic)] = 2 * ( ( precision[str(topic)] * recall[str(topic)] ) / ( precision[str(topic)] + recall[str(topic)] ) )
        accuracy[str(topic)] = 'NaN'

        recall_stemmed[str(topic)] = df[st_tp].sum(axis=0) / ( df[st_tp].sum(axis=0) + df[st_fn].sum(axis=0) )
        precision_stemmed[str(topic)] = df[st_tp].sum(axis=0) / ( df[st_tp].sum(axis=0) + df[st_fp].sum(axis=0) )
        f1score_stemmed[str(topic)] = 2 * ( ( precision_stemmed[str(topic)] * recall_stemmed[str(topic)] ) / ( precision_stemmed[str(topic)] + recall_stemmed[str(topic)] ) )
        accuracy_stemmed[str(topic)] = 'NaN'

    print("LENGTH OF THE DF\n\n")
    print(len(df))

    recall['Accuracy'] = recall_score(df['main_topic_label'], df['topic_label_dictionary'], average='weighted', sample_weight=None)
    precision['Accuracy'] = precision_score(df['main_topic_label'], df['topic_label_dictionary'], average='weighted', sample_weight=None)
    f1score['Accuracy'] = f1_score(df['main_topic_label'], df['topic_label_dictionary'], average='weighted', sample_weight=None)

    recall_stemmed['Accuracy'] = recall_score(df['main_topic_label'], df['stemmed_topic_label_dictionary'], average='weighted', sample_weight=None)
    precision_stemmed['Accuracy'] = precision_score(df['main_topic_label'], df['stemmed_topic_label_dictionary'], average='weighted', sample_weight=None)
    f1score_stemmed['Accuracy'] = f1_score(df['main_topic_label'], df['stemmed_topic_label_dictionary'], average='weighted', sample_weight=None)



    for tp, fp, fn, st_tp, st_fp, st_fn, topic in zip(true_positives_freq, false_positives_freq, false_negatives_freq, true_positives_st_freq, false_positives_st_freq, false_negatives_st_freq, topics) :

        recall_freq[str(topic)] = df[tp].sum(axis=0) / ( df[tp].sum(axis=0) + df[fn].sum(axis=0) )
        precision_freq[str(topic)] = df[tp].sum(axis=0) / ( df[tp].sum(axis=0) + df[fp].sum(axis=0) )
        f1score_freq[str(topic)] = 2 * ( ( precision[str(topic)] * recall[str(topic)] ) / ( precision[str(topic)] + recall[str(topic)] ) )
        accuracy_freq[str(topic)] = 'NaN'

        recall_stemmed_freq[str(topic)] = df[st_tp].sum(axis=0) / ( df[st_tp].sum(axis=0) + df[st_fn].sum(axis=0) )
        precision_stemmed_freq[str(topic)] = df[st_tp].sum(axis=0) / ( df[st_tp].sum(axis=0) + df[st_fp].sum(axis=0) )
        f1score_stemmed_freq[str(topic)] = 2 * ( ( precision_stemmed[str(topic)] * recall_stemmed[str(topic)] ) / ( precision_stemmed[str(topic)] + recall_stemmed[str(topic)] ) )
        accuracy_stemmed_freq[str(topic)] = 'NaN'

  #  print("LENGTH OF THE DF\n\n")
  #  print(len(df))

    recall_freq['Accuracy'] = recall_score(df['main_topic_label'], df['topic_label_dictionary-freq'], average='weighted', sample_weight=None)
    precision_freq['Accuracy'] = precision_score(df['main_topic_label'], df['topic_label_dictionary-freq'], average='weighted', sample_weight=None)
    f1score_freq['Accuracy'] = f1_score(df['main_topic_label'], df['topic_label_dictionary-freq'], average='weighted', sample_weight=None)

    recall_stemmed_freq['Accuracy'] = recall_score(df['main_topic_label'], df['stemmed_topic_label_dictionary-freq'], average='weighted', sample_weight=None)
    precision_stemmed_freq['Accuracy'] = precision_score(df['main_topic_label'], df['stemmed_topic_label_dictionary-freq'], average='weighted', sample_weight=None)
    f1score_stemmed_freq['Accuracy'] = f1_score(df['main_topic_label'], df['stemmed_topic_label_dictionary-freq'], average='weighted', sample_weight=None)


    return recall, precision, f1score, recall_stemmed, precision_stemmed, f1score_stemmed, recall_freq, precision_freq, f1score_freq, recall_stemmed_freq, precision_stemmed_freq, f1score_stemmed_freq



def get_scores():
    df = get_data()
    topics = list(df['main_topic_label'].unique())

    recall, precision, f1score, recall_stemmed, precision_stemmed, f1score_stemmed, recall_freq, precision_freq, f1score_freq, recall_stemmed_freq, precision_stemmed_freq, f1score_stemmed_freq = get_recall_precision(topics)


    total = { k: [ precision[k] , recall[k], f1score[k]] for k in recall }
    total_freq = { k: [ precision_freq[k] , recall_freq[k], f1score_freq[k]] for k in recall_freq }

    total_stemmed =  { k: [ precision_stemmed[k],  recall_stemmed[k] , f1score_stemmed[k]  ] for k in recall_stemmed }

    fname = '{}precision_recall_f1score_dictionary_not_stemmed.json'.format(OUTPUT_PATH)

    with open(fname, mode='w') as fo:
        json.dump(total, fo)

    logging.info("Created file {} with precision, recall and f1scores & accuracy".format(fname))


    fname = '{}precision_recall_f1score_dictionary_not_stemmed_freq.json'.format(OUTPUT_PATH)

    with open(fname, mode='w') as fo:
        json.dump(total_freq, fo)

    logging.info("Created file {} with precision, recall and f1scores & accuracy FREQ".format(fname))

    fname_stemmed = '{}precision_recall_f1score_dictionary_stemmed.json'.format(OUTPUT_PATH)

    with open(fname_stemmed, mode='w') as fo:
        json.dump(total_stemmed, fo)

    logging.info("Created file {} with precision, recall and f1scores & accuracy".format(fname_stemmed))

    fname_stemmed = '{}precision_recall_f1score_dictionary_stemmed_freq.json'.format(OUTPUT_PATH)

    with open(fname_stemmed, mode='w') as fo:
        json.dump(total_stemmed_freq, fo)

    logging.info("Created file {} with precision, recall and f1scores & accuracy".format(fname_stemmed))


if __name__ == "__main__":

    logger = logging.getLogger()
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)

    get_scores()
