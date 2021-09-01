"""
# Name(s): Shahir Taj
# Course: CSCI 4000 - A
# Date: 08/31/2021

"""

import logging
import glob
import heapq
import textwrap
import tensorflow_hub as hub
import numpy as np
import pandas as pd
import tensorflow_text

NUM_CLOSEST_PARAGRAPHS = 5


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
    embeddings = embed(list(paragraphs.values()))

    # Domain-specific (Digital Ricoeur) training after initial embeddings?

    all_proximities = np.empty([len(embeddings), len(embeddings)])
    closest_paragraphs = {}

    # Create a matrix of the proximities between all paragraphs.
    for i, target_embedding in enumerate(embeddings):
        target_proximities = []
        for j, compared_embedding in enumerate(embeddings):
            all_proximities[i, j] = np.inner(target_embedding, compared_embedding)
            if i != j:
                target_proximities.append((list(paragraphs.keys())[j], all_proximities[i, j]))
        closest_paragraphs[list(paragraphs.keys())[i]] = heapq.nlargest(NUM_CLOSEST_PARAGRAPHS, target_proximities,
                                                                        key=lambda x: x[1])

    return all_proximities, closest_paragraphs


def get_article_proximities(article_lengths, paragraph_proximities):
    proximities = np.empty([len(article_lengths), len(article_lengths)])

    # Create a matrix of the average proximities between articles.
    start_row = 0
    for i, article_length in enumerate(article_lengths):
        start_col = 0
        for j, target_length in enumerate(article_lengths):
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


def write_txt(paragraphs, closest_paragraphs, output_filepath):
    with open(output_filepath, 'w') as f:
        for target_label, target_text in paragraphs.items():
            f.write(target_label + '\n')
            f.write(textwrap.fill(target_text) + '\n\n')
            for (compared_label, proximity) in closest_paragraphs[target_label]:
                f.write('\t%s - %s\n' % (compared_label, proximity))
                f.write('\t' + textwrap.fill(paragraphs[compared_label], subsequent_indent="\t") + '\n\n')
            f.write('\n')


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
    paragraph_proximities, closest_paragraphs = get_paragraph_proximities(embed, paragraphs)
    logging.info("Paragraph Proximities Calculated")
    article_proximities = get_article_proximities(articles.values(), paragraph_proximities)
    logging.info("Article Proximities Calculated")
    logging.info("Closest Paragraphs Found")

    paragraph_proximities_filepath = "output/paragraph_proximities.csv"
    write_csv(paragraphs.keys(), paragraph_proximities, paragraph_proximities_filepath)
    logging.info("Paragraph Proximities Exported")
    article_proximities_filepath = "output/article_proximities.csv"
    write_csv(articles.keys(), article_proximities, article_proximities_filepath)
    logging.info("Article Proximities Exported")
    closest_paragraphs_filepath = "output/closest_paragraphs.txt"
    write_txt(paragraphs, closest_paragraphs, closest_paragraphs_filepath)
    logging.info("Closest Paragraphs Exported")

    logging.info("Finished")


if __name__ == "__main__":
    main()
