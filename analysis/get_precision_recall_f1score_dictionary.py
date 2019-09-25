
import pandas as pd
import logging
import json

PATH_TO_DATA = '~/surfdrive/uva/projects/RPA_KeepingScore/data/'
FILENAME = 'RPA_and_Buschers_data_with_dictionaryscores.pkl'

OUTPUT_PATH ='../output/'

def get_data():
    df = pd.read_pickle(PATH_TO_DATA + FILENAME)
    return df

def get_recall_precision(topics, sample):

    true_positives = ["_tp " + str(i) for i in topics]
    false_positives = ["_fp " + str(i) for i in topics]
    false_negatives = ["_fn " + str(i) for i in topics]

    true_positives_st = ["st_tp " + str(i) for i in topics]
    false_positives_st = ["st_fp " + str(i) for i in topics]
    false_negatives_st = ["st_fn " + str(i) for i in topics]

    recall = {}
    precision = {}
    f1score = {}

    recall_stemmed = {}
    precision_stemmed = {}
    f1score_stemmed = {}

    df = get_data()

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

    for tp, fp, fn, st_tp, st_fp, st_fn, topic in zip(true_positives, false_positives, false_negatives, true_positives_st, false_positives_st, false_negatives_st, topics) :

        recall[str(topic)] = df[tp].sum(axis=0) / ( df[tp].sum(axis=0) + df[fn].sum(axis=0) )
        precision[str(topic)] = df[tp].sum(axis=0) / ( df[tp].sum(axis=0) + df[fp].sum(axis=0) )

        f1score[str(topic)] = 2 * ( ( precision[str(topic)] * recall[str(topic)] ) / ( precision[str(topic)] + recall[str(topic)] ) )

        recall_stemmed[str(topic)] = df[st_tp].sum(axis=0) / ( df[st_tp].sum(axis=0) + df[st_fn].sum(axis=0) )
        precision_stemmed[str(topic)] = df[st_tp].sum(axis=0) / ( df[st_tp].sum(axis=0) + df[st_fp].sum(axis=0) )
        f1score_stemmed[str(topic)] = 2 * ( ( precision_stemmed[str(topic)] * recall_stemmed[str(topic)] ) / ( precision_stemmed[str(topic)] + recall_stemmed[str(topic)] ) )


    recall['total'] = sum(recall.values()) / len(recall.values())
    precision['total'] = sum(precision.values()) / len(precision.values())
    f1score['total'] = sum(f1score.values()) / len(f1score.values())

    recall_stemmed['total'] = sum(recall_stemmed.values()) / len(recall_stemmed.values())
    precision_stemmed['total'] = sum(precision_stemmed.values()) / len(precision_stemmed.values())
    f1score_stemmed['total'] = sum(f1score_stemmed.values()) / len(f1score_stemmed.values())

    return recall, precision, f1score, recall_stemmed, precision_stemmed, f1score_stemmed


def get_scores(sample):
    df = get_data()
    topics = list(df['main_topic_label'].unique())

    recall, precision, f1score, recall_stemmed, precision_stemmed, f1score_stemmed = get_recall_precision(topics, sample==sample)

    total = { k: [ precision[k] , recall[k], f1score[k]] for k in recall }
    total_stemmed =  { k: [ precision_stemmed[k],  recall_stemmed[k] , f1score_stemmed[k] ] for k in recall_stemmed }

    fname = '{}precision_recall_f1score_dictionary_not_stemmed{}.json'.format(OUTPUT_PATH, sample)

    with open(fname, mode='w') as fo:
        json.dump(total, fo)

    logging.info("Created file {} with precision, recall and f1scores, for the following sample: {}".format(fname, sample))

    fname_stemmed = '{}precision_recall_f1score_dictionary_stemmed{}.json'.format(OUTPUT_PATH, sample)

    with open(fname_stemmed, mode='w') as fo:
        json.dump(total_stemmed, fo)

    logging.info("Created file {} with precision, recall and f1scores, for the following sample: {}".format(fname_stemmed, sample))


if __name__ == "__main__":

    logger = logging.getLogger()
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)

    get_scores(sample="totalsample")
    get_scores(sample="newspaper_sample_only")
    get_scores(sample="pq_sample_only")
    get_scores(sample="RPA_sample")
    get_scores(sample="Bjorns_sample")

def get_scores(sample):
    df = get_data()
    topics = list(df['main_topic_label'].unique())
    recall, precision, f1score, recall_stemmed, precision_stemmed, f1score_stemmed = get_recall_precision(topics, sample==sample)

    total = { k: [ precision[k] , recall[k], f1score[k]] for k in recall }
    total_stemmed =  { k: [ precision_stemmed[k],  recall_stemmed[k] , f1score_stemmed[k] ] for k in recall_stemmed }

    fname = '{}precision_recall_f1score_dictionary_not_stemmed{}.json'.format(OUTPUT_PATH, sample)

    with open(fname, mode='w') as fo:
        json.dump(total, fo)

    logging.info("Created file {} with precision, recall and f1scores, for the following sample: {}".format(fname, sample))

    fname_stemmed = '{}precision_recall_f1score_dictionary_stemmed{}.json'.format(OUTPUT_PATH, sample)

    with open(fname_stemmed, mode='w') as fo:
        json.dump(total_stemmed, fo)

    logging.info("Created file {} with precision, recall and f1scores, for the following sample: {}".format(fname_stemmed, sample))


if __name__ == "__main__":

    logger = logging.getLogger()
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)

    get_scores(sample="totalsample")
    get_scores(sample="newspaper_sample_only")
    get_scores(sample="pq_sample_only")
    get_scores(sample="RPA_sample")
    get_scores(sample="Bjorns_sample")
