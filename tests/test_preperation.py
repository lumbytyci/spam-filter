from spamfilter import prepare

from keras.preprocessing.text import Tokenizer


def test_text_tokenizer():
    sample_texts = [
        "This is a sample email",
        "This, must be another example text!"
    ]

    tokenizer = prepare.get_prepared_tokenizer(sample_texts)
    if isinstance(tokenizer, Tokenizer):
        assert tokenizer.word_index["this"] == 1
