import data_loader as dl
import prepare

import numpy as np
from keras.preprocessing.sequence import pad_sequences


def train():
    labels, texts = dl.load_data_from_file('../data/collections/spam-corpus')
    # word_embeddings = dl.load_word_embeddings_from_file(
    #    '../data/word-embeddings/glove.6B.100d.txt')

    tokenizer = prepare.get_prepared_tokenizer(texts)
    texts = tokenizer.texts_to_sequences(texts)

    texts = np.array(texts)
    texts = pad_sequences(texts, maxlen=100)

    labels = prepare.encode_labels(labels)
    labels = np.array(labels)


if __name__ == "__main__":
    train()
