"""
# Name(s): Shahir Taj
# Course: CSCI 4000 - A
# Date: 12/06/2020

"""

import logging
import glob
import tensorflow_hub as hub
import numpy as np
import tensorflow_text
import pandas as pd


def process_files(files):
    articles = {}
    paragraphs = {}

    # Create a dictionary mapping paragraph names to paragraph text.
    for file in files:
        article = get_article(file)
        articles[file[6:-4]] = len(article)
        for paragraph_index, paragraph in enumerate(article):
            paragraphs[file[6:9] + ", " + str(paragraph_index)] = paragraph

    return articles, paragraphs


def get_article(filename):
    article = []

    with open(filename, 'r') as f:
        data = f.read()
        # Separate paragraphs by two newlines.
        paragraphs = data.split('\n\n')
        for paragraph in paragraphs:
            lines = paragraph.split('\n')
            if len(lines) > 1:
                first_line = lines[0]
                # Check if a paragraph's first line is a header.
                if first_line[0].isdigit():
                    lines = lines[1:]
            # Create a list of all of the article's paragraphs.
            article.append(' '.join(lines))

    return article


def get_paragraph_proximities(embed, paragraphs):
    # Compute embeddings.
    embeddings = embed(list(paragraphs))

    # Domain-specific (Digital Ricoeur) training after initial embeddings?

    proximities = np.empty([len(embeddings), len(embeddings)])

    # Create a matrix of the proximities between all paragraphs.
    for i, target_embedding in enumerate(embeddings):
        for j, compared_embedding in enumerate(embeddings):
            proximities[i, j] = np.inner(target_embedding, compared_embedding)

    return proximities


def get_article_proximities(article_lengths, paragraph_proximities):
    proximities = np.empty([len(article_lengths), len(article_lengths)])

    # Create a matrix of the average proximities between articles.
    start_row = 0
    for i, article_length in enumerate(article_lengths):
        start_col = 0
        for j, target_length in enumerate(article_lengths):
            # Exclude or include diagonals in average calculations?
            proximities[i, j] = np.average(paragraph_proximities[start_row:start_row + article_length,
                                           start_col:start_col + target_length])
            start_col += target_length
        start_row += article_length

    return proximities


def write_csv(labels, data, output_filepath):
    dataframe = pd.DataFrame(data)
    dataframe.columns = labels
    dataframe.index = labels
    dataframe.to_csv(output_filepath)


def main():
    logging.basicConfig(filename='csci-4000-a.log', format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',
                        level=logging.INFO)
    logging.info("Started")

    # Read input files.
    input_files = []
    input_filepath = "input/*.txt"
    for infile in glob.glob(input_filepath):
        try:
            input_files.append(infile)
        except FileNotFoundError:
            logging.info("The file %s was not found.", infile)

    if not input_files:
        logging.info("No input files found in %s. Exiting.", input_filepath)
        exit()
    logging.info("Files Read")

    articles, paragraphs = process_files(input_files)
    logging.info("Files Processed")

    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3")
    paragraph_proximities = get_paragraph_proximities(embed, paragraphs.values())
    logging.info("Paragraph Proximities Calculated")
    article_proximities = get_article_proximities(articles.values(), paragraph_proximities)
    logging.info("Article Proximities Calculated")

    paragraph_proximities_filepath = "output/paragraph_proximities.csv"
    write_csv(paragraphs.keys(), paragraph_proximities, paragraph_proximities_filepath)
    logging.info("Paragraph Proximities Exported")
    article_proximities_filepath = "output/article_proximities.csv"
    write_csv(articles.keys(), article_proximities, article_proximities_filepath)
    logging.info("Article Proximities Exported")

    logging.info("Finished")


if __name__ == "__main__":
    main()
