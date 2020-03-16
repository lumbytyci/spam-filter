def decode_index(index: int) -> str:
    return {0: "ham", 1: "spam"}[index]


def probability_to_index(prediction: list) -> int:
    prob_to_int = list(map(round, prediction))
    return prob_to_int.index(max(prob_to_int))
