"""
# Name(s): Shahir Taj
# Course: CSCI 4000 - A
# Date: 10/21/2020

"""

import tensorflow_hub as hub
import numpy as np
import tensorflow_text
import glob


def split_file_into_list(file_name):
    # read the data from a file
    output = []
    with open(file_name, 'r') as f:
        data = f.read()
        # paragraphs are separated by two newlines
        paragraphs = data.split('\n\n')
        # each paragraph string within a file
        for paragraph in paragraphs:
            lines = paragraph.split('\n')
            # possible header
            if len(lines) > 1:
                first_line = lines[0]
                if first_line[0].isdigit():
                    new_list = lines[1:]
                    output.append(' '.join(new_list))
    return output


def calculate_proximities(embed, articles, num_paragraphs):
    paragraph_proximities = np.empty((num_paragraphs, num_paragraphs))

    for article1 in articles:
        for j, target_paragraph in enumerate(article1):
            print(target_paragraph)
            for article2 in articles:
                for m, compared_paragraph in enumerate(article2):
                    proximity = compare_paragraphs(embed, target_paragraph, compared_paragraph)
                    paragraph_proximities[j, m] = proximity

    return paragraph_proximities


def compare_paragraphs(embed, segment_one, segment_two):
    # Compute embeddings.
    segment_one_result = embed(segment_one)
    segment_two_result = embed(segment_two)

    # Domain-specific (Digital Ricoeur) training after initial embeddings

    # Compute similarity matrix. Higher score indicates greater similarity.
    similarity_matrix = np.inner(segment_one_result, segment_two_result)

    return similarity_matrix


def main():
    # read input files
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

    articles = []
    num_paragraphs = 0
    for file in input_files:
        article = split_file_into_list(file)
        articles.append(article)
        num_paragraphs += len(article)

    print(articles)
    print(num_paragraphs)

    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3")
    paragraph_proximities = calculate_proximities(embed, articles, num_paragraphs)

    print(paragraph_proximities)


if __name__ == "__main__":
    main()
