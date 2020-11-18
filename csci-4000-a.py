"""
# Name(s): Shahir Taj
# Course: CSCI 4000 - A
# Date: 11/18/2020

"""

import tensorflow_hub as hub
import numpy as np
import tensorflow_text
import glob
import pandas as pd


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


def calculate_proximities(embed, articles, num_paragraphs):
    proximities = np.empty((num_paragraphs, num_paragraphs))

    # Create a matrix of the proximities between all paragraphs.
    for i, (target_paragraph_key, target_paragraph) in enumerate(articles.items()):
        for j, (compared_paragraph_key, compared_paragraph) in enumerate(articles.items()):
            # Calculate the proximity between each paragraph in all articles.
            proximity = compare_paragraphs(embed, target_paragraph, compared_paragraph)
            proximities[i, j] = proximity

    return proximities


def compare_paragraphs(embed, segment_one, segment_two):
    # Compute embeddings.
    segment_one_result = embed(segment_one)
    segment_two_result = embed(segment_two)

    # Domain-specific (Digital Ricoeur) training after initial embeddings.

    # Compute similarity matrix. Higher score indicates greater similarity.
    similarity_matrix = np.inner(segment_one_result, segment_two_result)

    return similarity_matrix


def write_csv(data, labels):
    dataframe = pd.DataFrame(data)
    dataframe.columns = labels
    dataframe.index = labels
    dataframe.to_csv("output/paragraph_proximities.csv")


def main():
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

    articles = {}
    paragraph_labels = []
    num_paragraphs = 0
    # Create a dictionary mapping paragraph locations to paragraph text.
    for file in input_files:
        article = get_article(file)
        for j, paragraph in enumerate(article):
            articles[(file[6:9], j)] = paragraph
            paragraph_labels.append(file[6:9] + ", " + str(j))
            num_paragraphs += 1

    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3")
    paragraph_proximities = calculate_proximities(embed, articles, num_paragraphs)

    write_csv(paragraph_proximities, paragraph_labels)


if __name__ == "__main__":
    main()
