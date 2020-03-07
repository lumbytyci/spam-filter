import pytest

from spamfilter import data_loader

spam_data_path = 'data/collections/spam-corpus'
word_embeddings_path = 'data/word-embeddings/glove.6B.100d.txt'


def test_loading_spam_data_from_file():
    labels, emails = data_loader.load_data_from_file(spam_data_path)
    assert labels is not None
    assert emails is not None


def test_loaded_data_integrity_by_comparing_len():
    labels, emails = data_loader.load_data_from_file(spam_data_path)
    assert len(labels) == len(emails)


@pytest.mark.skip(reason="word-embedding file is huge")
def test_loading_word_embeddings_from_file():
    word_embeddings = data_loader.load_word_embeddings_from_file(
        word_embeddings_path)
    assert word_embeddings is not None
    assert len(word_embeddings) == 400000
