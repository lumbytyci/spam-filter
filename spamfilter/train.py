import data_loader as dl
import prepare
import model

import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint
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

    seq_model = model.get_compiled_model(embeddings_matrix, 100)

    checkpoint_file_path = "../checkpoints/weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
    checkpoint = ModelCheckpoint(
        checkpoint_file_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    seq_model.fit(texts_train, labels_train, validation_data=(texts_test, labels_test),
                  batch_size=32, epochs=16,
                  callbacks=callbacks_list,
                  verbose=1)

    performance = seq_model.evaluate(texts_test, labels_test)
    print(performance)


if __name__ == "__main__":
    train()
