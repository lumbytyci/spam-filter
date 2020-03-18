import hashlib
import time

pow_header = {
    'email': 'john.doe@example.com',
    'timestamp': int(time.time()),
    'zero_bits': 4,
    'nonce': '4XvjKyMnTlHf81K',
    'seed': 0
}


def construct_header_from_input(header: dict):
    sender_email = str(input("Sender email: "))
    try:
        difficulty = int(input("Difficulty (n-zero-bits): "))
    except ValueError:
        difficulty = 8
        print(f"Invalid difficulty level, defaulting to {difficulty}.")

    pow_header['email'] = sender_email
    pow_header['zero_bits'] = difficulty


def forge_header(header: dict) -> str:
    return ':'.join(map(str, header.values()))


def matches_difficulty(digest: int, difficulty: int) -> bool:
    return not (digest & (pow(2, difficulty) - 1))


def do_work(header: dict) -> str:
    h = hashlib.sha256(forge_header(header).encode())

    while not matches_difficulty(int.from_bytes(h.digest(), 'little'), header['zero_bits']):
        pow_header['seed'] += 1
        h = hashlib.sha256(forge_header(header).encode())

    return h.hexdigest()


if __name__ == "__main__":
    construct_header_from_input(pow_header)
    print(do_work(pow_header))
    print(pow_header)
