

filenames = ['../data/raw/embeddings/sentences_parliamentary_questions.txt', '../data/raw/embeddings/uniekezinnen_2000-01-01_2018-12-31.txt']
with open('../data/raw/embeddings/political_news_corpus.txt', 'w') as outfile:
    for fname in filenames:
        with open(fname) as infile:
            for line in infile:
                outfile.write(line)
