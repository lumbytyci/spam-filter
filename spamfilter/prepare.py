import numpy as np
from keras.preprocessing.text import Tokenizer


def get_prepared_tokenizer(email_texts: list) -> Tokenizer:
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(email_texts)

    return tokenizer


def map_embeddings_to_word_index(embeddings: dict, word_index: dict):

    mapped_embeddings = np.zeros((len(word_index) + 1, 100))
    for word, index in word_index.items():
        emb_word_vector = embeddings.get(word)
        if emb_word_vector is not None:
            mapped_embeddings[index] = emb_word_vector

    return mapped_embeddings


def encode_labels(labels: list) -> list:
    encoded_labels = []
    for label in labels:
        if label == "ham":
            encoded_labels.append([1.0, 0.0])
        else:
            encoded_labels.append([0.0, 1.0])

    return encoded_labels


if __name__ == "__main__":
    test_emails = [
        "This is a sample email!",
        "This is another sample email!"
    ]

    print(get_prepared_tokenizer(test_emails).word_index)
