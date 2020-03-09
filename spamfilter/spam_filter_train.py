import prepare
import model
import train
import data_loader as dl

import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences


def spam_filter():
    config = prepare.load_model_config('../config.yml')

    labels, texts = dl.load_data_from_file('../data/collections/spam-corpus')
    word_embeddings = dl.load_word_embeddings_from_file(
        '../data/word-embeddings/glove.6B.100d.txt')

    tokenizer = prepare.get_prepared_tokenizer(texts)
    texts = tokenizer.texts_to_sequences(texts)

    texts = np.array(texts)
    texts = pad_sequences(texts, maxlen=config['dataset']['max_seq_len'])

    labels = prepare.encode_labels(labels)
    labels = np.array(labels)

    texts_train, texts_test, labels_train, labels_test = train_test_split(
        texts, labels, random_state=config['dataset']['random_state'],
        test_size=config['dataset']['test_size'])

    embeddings_matrix = prepare.map_embeddings_to_word_index(
        word_embeddings, tokenizer.word_index)

    seq_model = model.get_compiled_model(embeddings_matrix, config)
    seq_model.summary()

    train.train_model(seq_model, config, texts_train,
                      labels_train, texts_test, labels_test)


if __name__ == "__main__":
    spam_filter()
