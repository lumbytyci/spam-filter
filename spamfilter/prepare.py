from keras.preprocessing.text import Tokenizer


def get_prepared_tokenizer(email_texts: list) -> Tokenizer:
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(email_texts)

    return tokenizer


if __name__ == "__main__":
    test_emails = [
        "This is a sample email!",
        "This is another sample email!"
    ]

    print(get_prepared_tokenizer(test_emails).word_index)
