from keras.models import Sequential
from keras.layers import LSTM, Embedding, Dense, Dropout


def get_compiled_model(embeddings_matrix: dict, num_lstm_units):
    model = Sequential()
    embedding_layer = Embedding(len(embeddings_matrix),
                                100,
                                trainable=False,
                                input_length=100,
                                weights=[embeddings_matrix])

    lstm_layer = LSTM(num_lstm_units, recurrent_dropout=0.2)

    dropout_layer = Dropout(0.3)

    dense_layer = Dense(2, activation="softmax")

    model.add(embedding_layer)
    model.add(lstm_layer)
    model.add(dropout_layer)
    model.add(dense_layer)

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    model.summary()

    return model
