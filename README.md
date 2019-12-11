# RPA
This repo attempts to compare different approaches to detect topics in political and media content, and compare their effectiveness.

---
This repo contains the following elements:

-  Data preparation
    - Prep and combine data from coders, newspapers and parlementairly questions
- Evaluation of classification accuracy of these documents:
	- Dictionary approach, using word lists constructed in [here](https://www.almendron.com/tribuna/wp-content/uploads/2017/05/CAP2013v2.pdf)
	- 'Classic' Supervised Machine Learning
	- CNN models using pretrained word embedding models trained on Dutch news and parliamentary questions.

We extend our dataset with the newspaper dataset reported here: https://journals.sagepub.com/doi/10.1177/0002716215569441

---

# Folder, scripts and datasets

## `helpers/`

In the folder `helpers/`, you find helpers to parse the parliamentary questions and scripts to prep the annotated data and merge with the original text files (newspapers + parlementairly questions)

### parser

1.  `kamervragen_parser.py` : use this script to parse the parliamentary questions

### prep data

1. `prep_annotated_data.py` : this scrips prepares the raw file with annotated data (`PRA_coding.csv`).

2.  `prep_newspaper_data.py`  :

dependencies: uses `prep_annotated_data.py` to get cleaned data, and merges the annotated data with the text of the newspaper articles. creates the file `data/raw/kamervragen_merged_with_annotated`

3.  `prep_kamervragen_data.py` : this script merges the annotated data with the parsed questions belonging to parliamentary questions.

dependencies: uses `prep_annotated_data.py` to get cleaned data. Creates the file `data/raw/VK_TEL_merged_with_annotated.pkl`

4. `get_dictionary_scores.py`
dependencies: `VK_TEL_merged_with_annotated.pkl` & `kamervragen_merged_with_annotated`, merge them and apply the CAP dictionary. It outputs the file: `RPA_data_with_dictionaryscores.pkl`

5. `preproces.py`
dependencies: Takes the file `../data/intermediate/RPA_data_with_dictionaryscores.pkl` and adds a cleaned, preprocessed text column.


### analysis

#### run_classifier.py

## Commands:

run parser for kamervragen:

```
python3 helpers/kamervragen_parser.py --data_path ~/Dropbox/kamervragen-xml/  --output ~/surfdrive/uva/projects/RPA_KeepingScore/data/

```
