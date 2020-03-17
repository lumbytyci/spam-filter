from spamfilter import utils


def test_probability_to_index():
    assert utils.probability_to_index([0.99, 0.002]) == 0
    assert utils.probability_to_index([0.09, 0.91]) == 1
