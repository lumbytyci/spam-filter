import prepare
import model
import train
import data_loader as dl

import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences

config = prepare.load_model_config('../config.yml')
dataset_path = '../data/collections/spam-corpus'
word_emb_path = '../data/word-embeddings/glove.6B.100d.txt'


def get_prediction(seq_model, tokenizer, text):
    seq = tokenizer.texts_to_sequences([text])
    seq = pad_sequences(seq, maxlen=100)
    prediction = seq_model.predict(seq)[0]
    print(prediction)


def load_weights_from_file(path: str, seq_model, X_test, y_test):
    seq_model.load_weights(path)
    print(seq_model.evaluate(X_test, y_test, verbose=2))


def spam_filter(train_mode: bool):

    labels, texts = dl.load_data_from_file(dataset_path)
    word_embeddings = dl.load_word_embeddings_from_file(word_emb_path)

    tokenizer = prepare.get_prepared_tokenizer(texts)
    texts = tokenizer.texts_to_sequences(texts)

    texts = np.array(texts)
    texts = pad_sequences(texts, maxlen=config['dataset']['max_seq_len'])

    labels = prepare.encode_labels(labels)
    labels = np.array(labels)

    texts_train, texts_test, labels_train, labels_test = train_test_split(
        texts, labels,
        random_state=config['dataset']['random_state'],
        test_size=config['dataset']['test_size'])

    embeddings_matrix = prepare.map_embeddings_to_word_index(
        word_embeddings, tokenizer.word_index)

    seq_model = model.get_compiled_model(embeddings_matrix, config)
    seq_model.summary()

    if train_mode:
        train.train_model(seq_model, config, texts_train,
                          labels_train, texts_test, labels_test)
    else:
        load_weights_from_file(
            '../checkpoints/weights-improvement-12-0.98.hdf5', seq_model, texts_test, labels_test)


if __name__ == "__main__":
    spam_filter(train_mode=False)
