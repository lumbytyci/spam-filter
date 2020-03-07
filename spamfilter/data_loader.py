import numpy as np


def load_data_from_file(path: str) -> tuple:
    """Load data from spam collection text file

    Parameters:
    path (str): Path to text file

    Returns:
    tuple: Returns tuple with labels and emails
    """

    with open(path, 'r') as spam_collection_file:
        labels = []
        emails = []
        for line in spam_collection_file:
            label, email = line.strip().split(maxsplit=1)
            labels.append(label.strip())
            emails.append(email.strip())

    return labels, emails


def load_word_embeddings_from_file(path: str) -> dict:
    """Load pre-trained word embeddings from text file

    Parameters:
    path (str): Path to text file

    Return:
    dict: Returns dict with key -> word, val -> vector (list)
    """

    word_embeddings = {}
    with open(path, 'r', encoding='utf8') as word_embeddings_file:
        for line in word_embeddings_file:
            word, *vector = line.split()
            vector = np.asarray(vector, dtype='float32')
            word_embeddings[word] = vector

    return word_embeddings


if __name__ == "__main__":
    labels, emails = load_data_from_file('../data/collections/spam-corpus')
    print(len(max(emails)))
    print("Reading word embeddings...")
    word_embeddings = load_word_embeddings_from_file(
        '../data/word-embeddings/glove.6B.100d.txt')
    print("Read {} word embeddings".format(len(word_embeddings)))
