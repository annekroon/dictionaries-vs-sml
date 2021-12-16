# Classifying topics and frames

This repo attempts to compare different approaches to detect topics in political and media content, and compare their effectiveness. 

---

This repo contains the following elements:

-  Data preparation
    - Prep and combine data from coders, newspapers and parlementairly questions
- Evaluation of classification accuracy of these documents:
	- Dictionary approach, using word lists constructed in [here](https://www.almendron.com/tribuna/wp-content/uploads/2017/05/CAP2013v2.pdf)
	- Supervised Machine Learning
	- Machine learning using vectorizers based on word embedding models trained on Dutch news and parliamentary questions.

---


# Folders and code

## `helpers/`

In the folder `helpers/`, you find helpers to parse the parliamentary questions and scripts to prep the annotated data and merge with the original text files (newspapers + parlementairly questions)

### parser

1.  `kamervragen_parser.py` : use this script to parse the parliamentary questions

### prep data

1. `prep_annotated_data.py` : this scrips prepares the raw file with annotated data (`PRA_coding.csv`).

2.  `prep_newspaper_data.py`  :

Dependencies: uses `prep_annotated_data.py` to get cleaned data, and merges the annotated data with the text of the newspaper articles. creates the file `data/raw/kamervragen_merged_with_annotated`

3.  `prep_kamervragen_data.py` : this script merges the annotated data with the parsed questions belonging to parliamentary questions.

Dependencies: uses `prep_annotated_data.py` to get cleaned data. Creates the file `data/raw/VK_TEL_merged_with_annotated.pkl`

4. `get_dictionary_scores.py`
Dependencies: `VK_TEL_merged_with_annotated.pkl` & `kamervragen_merged_with_annotated`, merge them and apply the CAP dictionary. It outputs the file: `RPA_data_with_dictionaryscores.pkl`

5. `preproces.py`
Dependencies: Takes the file `../data/intermediate/RPA_data_with_dictionaryscores.pkl` and adds a cleaned, preprocessed text column.


## `analysis/`

Files in this folder run the analysis and create tables and figures.
All files starting with 'F' were created for the analysis of the frame variables, all files starting with 'T' for the analysis about policy issues (i.e., topics).

## `model training/`

Files in this folder were used to create the corpus + train the embedding model.


## `resources/`

Access to CAP dictionary.


## Commands:

run parser for kamervragen:

```
python3 helpers/kamervragen_parser.py --data_path ~/Dropbox/kamervragen-xml/  --output ~/surfdrive/uva/projects/RPA_KeepingScore/data/

```

get all datasets used for this study (both raw + intermediate + word embedding file)

```
python3 download_datasets.py

```

# Data

All data for model training can be found here: `data/raw/embedding`.

`data/raw/embedding/uniekezinnen_2000-01-01_2018-12-31.txt` contains the sentences from news articles
`data/raw/embedding/sentences_parliamentary_questions.txt` contains the sentences from parliamentary questions.
`data/raw/embedding/political_news_corpus.txt' combined the news + political data and is used for final model training.
