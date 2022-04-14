"""BERT-Driven Concept Detection in Philosophical Corpora

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
"""

import logging

import glob

import heapq

import textwrap

import tensorflow_hub as hub

import numpy as np

import pandas as pd

import tensorflow_text


NAME_START_INDEX = 6

ARTICLE_NAME_END_INDEX = -4

PARAGRAPH_NAME_END_INDEX = 9

NUM_CLOSEST_PARAGRAPHS = 5


def process_files(input_files):
    """Read the input files and return a dictionary of articles and a dictionary of paragraphs.

    Parameters
    ----------
    input_files : list
        The list of input file names

    Returns
    -------
    dict
        a dictionary of articles in the files
    dict
        a dictionary of paragraphs in the articles
    """
    articles = {}
    paragraphs = {}

    # Create a dictionary mapping article names to article length.
    for file in input_files:
        article = get_article(file)
        articles[file[NAME_START_INDEX:ARTICLE_NAME_END_INDEX]] = len(article)
        # Create a dictionary mapping paragraph names to paragraph text.
        for paragraph_index, paragraph in enumerate(article):
            paragraphs[file[NAME_START_INDEX:PARAGRAPH_NAME_END_INDEX] + ", " + str(paragraph_index)] = paragraph

    return articles, paragraphs


def get_article(file):
    """Split an article file into paragraphs and return a list.

    Parameters
    ----------
    file : str
        The path of the article file

    Returns
    -------
    list
        a list of paragraphs in the article
    """
    article = []

    with open(file, 'r') as f:
        data = f.read()
        # Separate paragraphs by two newlines.
        paragraphs = data.split('\n\n')
        for paragraph in paragraphs:
            lines = paragraph.split('\n')
            if len(lines) > 1:
                first_line = lines[0]
                # Check if the paragraph's first line is a header.
                if first_line[0].isdigit():
                    lines = lines[1:]
            article.append(' '.join(lines))

    return article


def get_paragraph_proximities(embed, paragraphs):
    """Calculate the proximities between all paragraphs and return them in a matrix and a dictionary of the closest
    paragraphs to each paragraph.

    Parameters
    ----------
    paragraphs : dict
        A dictionary of paragraphs in the articles

    Returns
    -------
    ndarray
        a matrix of the proximities between all paragraphs
    dict
        a dictionary of the closest paragraphs to each paragraph
    """
    # Compute embeddings.
    embeddings = embed(list(paragraphs.values()))

    # TODO: Add domain-specific training after initial embeddings?

    paragraph_proximities = np.empty([len(embeddings), len(embeddings)])
    closest_paragraphs = {}

    # Create a matrix of the proximities between all paragraphs.
    for i, target_embedding in enumerate(embeddings):
        target_proximities = []
        for j, compared_embedding in enumerate(embeddings):
            paragraph_proximities[i, j] = np.inner(target_embedding, compared_embedding)
            if i != j:
                target_proximities.append((list(paragraphs.keys())[j], paragraph_proximities[i, j]))
        # Find and store the paragraphs with the highest proximities to each paragraph.
        closest_paragraphs[list(paragraphs.keys())[i]] = heapq.nlargest(NUM_CLOSEST_PARAGRAPHS, target_proximities,
                                                                        key=lambda x: x[1])

    return paragraph_proximities, closest_paragraphs


def get_article_proximities(article_lengths, paragraph_proximities):
    """Calculate the average proximities between all articles and return them in a matrix.

    Parameters
    ----------
    article_lengths : list
        The lengths of all articles
    paragraph_proximities : ndarray
        A matrix of the proximities between all paragraphs

    Returns
    -------
    ndarray
        a matrix of the average proximities between all articles
    """
    proximities = np.empty([len(article_lengths), len(article_lengths)])

    start_row = 0
    for i, article_length in enumerate(article_lengths):
        start_col = 0
        for j, target_length in enumerate(article_lengths):
            proximities[i, j] = np.average(paragraph_proximities[start_row:start_row + article_length,
                                           start_col:start_col + target_length])
            start_col += target_length
        start_row += article_length

    return proximities


def write_csv(labels, data, output_file):
    """Write and save the proximities between all paragraphs or the average proximities between all articles in a .csv file.

    Parameters
    ----------
    labels : list
        The list of names of all paragraphs or articles
    data : ndarray
        The matrix of proximities between all paragraphs or avergage proximities between all articles
    output_file : str
        The path of the output file
    """
    dataframe = pd.DataFrame(data)
    dataframe.columns = labels
    dataframe.index = labels
    dataframe.to_csv(output_file)


def write_txt(paragraphs, closest_paragraphs, output_file):
    """Write and save the closest paragraphs to each paragraph in a .txt file.

    Parameters
    ----------
    paragraphs : dict
        A dictionary of paragraphs in the articles
    closest_paragraphs : dict
        A dictionary of the closest paragraphs to each paragraph
    output_file : str
        The path of the output file
    """
    with open(output_file, 'w') as f:
        for target_label, target_text in paragraphs.items():
            f.write(target_label + '\n')
            f.write(textwrap.fill(target_text) + '\n\n')
            for (compared_label, proximity) in closest_paragraphs[target_label]:
                f.write('\t%s - %s\n' % (compared_label, proximity))
                f.write('\t' + textwrap.fill(paragraphs[compared_label], subsequent_indent="\t") + '\n\n')
            f.write('\n')


def main():
    logging.basicConfig(filename='bert_concept_detection.log', format='%(asctime)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
    logging.info("Started")

    # Read the input files.
    input_files = []
    input_filepath = "input/*.txt"
    for file in glob.glob(input_filepath):
        try:
            input_files.append(file)
        except FileNotFoundError:
            logging.info("The file %s was not found.", file)

    if not input_files:
        logging.info("No input files found in %s. Exiting.", input_filepath)
        exit()
    logging.info("Files Read")

    # Process the input files.
    articles, paragraphs = process_files(input_files)
    logging.info("Files Processed")

    # Calculate the proximities between all paragraphs and the average proximities between all articles.
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3")

    paragraph_proximities, closest_paragraphs = get_paragraph_proximities(embed, paragraphs)
    logging.info("Paragraph Proximities Calculated")
    article_proximities = get_article_proximities(articles.values(), paragraph_proximities)
    logging.info("Article Proximities Calculated")
    logging.info("Closest Paragraphs Found")

    # Export the proximities and the closest paragraphs to each paragraph.
    paragraph_proximities_file = "output/paragraph_proximities.csv"
    write_csv(paragraphs.keys(), paragraph_proximities, paragraph_proximities_file)
    logging.info("Paragraph Proximities Exported")
    article_proximities_file = "output/article_proximities.csv"
    write_csv(articles.keys(), article_proximities, article_proximities_file)
    logging.info("Article Proximities Exported")
    closest_paragraphs_file = "output/closest_paragraphs.txt"
    write_txt(paragraphs, closest_paragraphs, closest_paragraphs_file)
    logging.info("Closest Paragraphs Exported")

    logging.info("Finished")


if __name__ == "__main__":
    main()
