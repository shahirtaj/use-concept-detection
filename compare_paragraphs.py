import tensorflow_hub as hub
import numpy as np
import tensorflow_text


def compare_paragraphs(segment_one, segment_two):
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3")

    # Compute embeddings.
    segment_one_result = embed(segment_one)
    segment_two_result = embed(segment_two)

    # Domain-specific (Digital Ricoeur) training after initial embeddings

    # Compute similarity matrix. Higher score indicates greater similarity.
    similarity_matrix = np.inner(segment_one_result, segment_two_result)

    print(similarity_matrix)
    return similarity_matrix


paragraph_one = ["What do we know of this death? In part, it is a death that does not know itself; it is the living "
                 "death of those who believe themselves living. But in part, also, it is a death that is suffered: "
                 "'When the commandment came, sin revived and I died' (Rom. 7:9-10). What shall we say? Without doubt "
                 "it is legitimate to compare this death that is suffered with the experience of division and conflict "
                 "described in the pericope of the Epistle to the Romans (7:14—19), which follows the dialectic of sin "
                 "and the law reported above. Death, then, is the actualized dualism of the Spirit and the flesh."]
paragraph_two = ["Wherefore the law is holy, and the commandment holy, and just, and good, Was then that which is good "
                 "made death unto me? God forbid. But sin, that it might appear sin, working death in me by that which "
                 "is good; that sin by the commandment might become exceeding sinful."]

# We could also do some kind of comparison between the individual sentences in the query and segment.

compare_paragraphs(paragraph_one, paragraph_two)
