# BERT-Driven Concept Detection in Philosophical Corpora

This script accepts a corpora of text files (.txt) containing Aristotle's entries in the Stanford Encyclopedia of
Philosophy as input, uses Google's Universal Sentence Encoder algorithm to embed each paragraph, and generates the
proximities, or semantic textual similarity, between each paragraph and each article as output in comma-separated value
files (.csv). It also generates the five paragraphs with the highest proximities to each paragraph as output in a text
file (.txt).

This file contains the following functions:
    * process_files - return a dictionary of articles and a dictionary of paragraphs
    * get_article - return a list of paragraphs in an article
    * get_paragraph_proximities - return a matrix of the proximities between all paragraphs and a dictionary of the
                                  closest paragraphs to each paragraph
    * get_article_proximities - return a matrix of the avergage proximities between all articles
    * write_csv - save the proximities between all paragraphs or the average proximities between all articles in a .csv
                  file
    * write_txt - save the closest paragraphs to each paragraph in a .txt file
    * main - the main function of the script

## TODO

Add domain-specific training after initial embeddings

## License

Distributed under the MIT license. See ``LICENSE`` for more information.
