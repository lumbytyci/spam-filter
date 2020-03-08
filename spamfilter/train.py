import data_loader as dl
import prepare
import model

import numpy as np
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


def train():
    labels, texts = dl.load_data_from_file('../data/collections/spam-corpus')
    word_embeddings = dl.load_word_embeddings_from_file(
        '../data/word-embeddings/glove.6B.100d.txt')

    tokenizer = prepare.get_prepared_tokenizer(texts)
    texts = tokenizer.texts_to_sequences(texts)

    texts = np.array(texts)
    texts = pad_sequences(texts, maxlen=100)

    labels = prepare.encode_labels(labels)
    labels = np.array(labels)

    texts_train, texts_test, labels_train, labels_test = train_test_split(
        texts, labels, random_state=10, test_size=0.25)

    embeddings_matrix = prepare.map_embeddings_to_word_index(
        word_embeddings, tokenizer.word_index)

    model.get_compiled_model(embeddings_matrix, 100)


if __name__ == "__main__":
    train()
