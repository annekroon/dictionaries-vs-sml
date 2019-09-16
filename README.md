# RPA
This repo attempts to compare different approaches to detect topics in political and media content, and compare their effectiveness.

---
This repo contains the following elements:

-  Data preparation
    - Prep and combine data from coders, newspapers and parlementairly questions
- Evaluation of classification accuracy of these documents:
	- Dictionary approach, using word lists constructed in [here](https://www.almendron.com/tribuna/wp-content/uploads/2017/05/CAP2013v2.pdf)
	- Machine Learning

In doing so, we compare the here-trained models with pre-trained word embedding models on Dutch corpora (i.e., the COW model and a FastText model trained on Wikipedia data, available here: https://github.com/clips/dutchembeddings).

---
## Python scripts:

#### run_classifier.py
