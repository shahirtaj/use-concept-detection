"""
# Name(s): Shahir Taj
# Course: CSCI 4000 - A
# Date: 11/19/2020

"""

import logging
import glob
import tensorflow_hub as hub
import numpy as np
import tensorflow_text
import pandas as pd


def get_paragraphs(files):
    paragraphs = {}
    # Create a dictionary mapping paragraph names to paragraph text.
    for file in files:
        article = get_article(file)
        for paragraph_index, paragraph in enumerate(article):
            paragraphs[file[6:9] + ", " + str(paragraph_index)] = paragraph

    return paragraphs


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


def calculate_proximities(embed, paragraphs):
    # Compute embeddings.
    embeddings = embed(list(paragraphs.values()))

    # Domain-specific (Digital Ricoeur) training after initial embeddings.

    proximities = np.empty([len(embeddings), len(embeddings)])

    # Create a matrix of the proximities between all paragraphs.
    num_comparisons = 0
    for i, target_paragraph in enumerate(embeddings):
        for j, compared_paragraph in enumerate(embeddings):
            proximities[i, j] = np.inner(target_paragraph, compared_paragraph)

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
            print("The file " + infile + " was not found.")

    if not input_files:
        print("No input files found in " + input_filepath + ". Exiting.")
        exit()
    logging.info("Files Read")

    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3")
    paragraphs = get_paragraphs(input_files)
    paragraph_proximities = calculate_proximities(embed, paragraphs)
    logging.info("Proximities Calculated")

    output_filepath = "output/paragraph_proximities.csv"
    write_csv(paragraphs.keys(), paragraph_proximities, output_filepath)
    logging.info("Finished")


if __name__ == "__main__":
    main()
