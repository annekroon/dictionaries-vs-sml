

filenames = ['..data/raw/sentences_parliamentary_questions.txt, '..data/raw/file2.txt']
with open('..data/raw/political_news_corpus.txt', 'w') as outfile:
    for fname in filenames:
        with open(fname) as infile:
            for line in infile:
                outfile.write(line)
