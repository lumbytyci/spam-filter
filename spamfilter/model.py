import prepare

from keras.models import Sequential
from keras.layers import LSTM, Embedding, Dense, Dropout


def get_compiled_model(embeddings_matrix: dict, config: dict) -> Sequential:

    model = Sequential()
    embedding_layer = Embedding(len(embeddings_matrix),
                                config['embedding']['num_nodes'],
                                trainable=False,
                                input_length=config['embedding']['input_len'],
                                weights=[embeddings_matrix])

    lstm_layer = LSTM(config['lstm']['num_units'],
                      recurrent_dropout=config['lstm']['dropout'])

    dropout_layer = Dropout(config['dropout']['val'])

    dense_layer = Dense(2, activation="softmax")

    model.add(embedding_layer)
    model.add(lstm_layer)
    model.add(dropout_layer)
    model.add(dense_layer)

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    return model
