def decode_index(index: int) -> str:
    return {0: "ham", 1: "spam"}[index]


def probability_to_index(prediction: list) -> int:
    return 0 if prediction[0] > prediction[1] else 1
