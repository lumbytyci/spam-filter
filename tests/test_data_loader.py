from spamfilter import data_loader

spam_data_path = 'data/collections/spam-corpus'


def test_loading_data_from_file():
    labels, emails = data_loader.load_data_from_file(spam_data_path)
    assert labels is not None
    assert emails is not None


def test_loaded_data_integrity_by_comparing_len():
    labels, emails = data_loader.load_data_from_file(spam_data_path)
    assert len(labels) == len(emails)
