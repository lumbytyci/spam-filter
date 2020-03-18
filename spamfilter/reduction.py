import hashlib
import time

pow_header = {
    'email': 'john.doe@example.com',
    'timestamp': int(time.time()),
    'zero_bits': 4,
    'nonce': '4XvjKyMnTlHf81K',
    'seed': 0
}


def forge_header(header: dict) -> str:
    return ':'.join(map(str, header.values()))


def matches_difficulty(digest: int, difficulty: int) -> bool:
    return not (digest & (pow(2, difficulty) - 1))


def do_work(header: dict, difficulty: int) -> str:
    print(forge_header(header))
    h = hashlib.sha256(forge_header(header).encode())

    while not matches_difficulty(int.from_bytes(h.digest(), 'little'), difficulty):
        pow_header['seed'] += 1
        h = hashlib.sha256(forge_header(header).encode())


if __name__ == "__main__":
    do_work(pow_header, 16)
    print(pow_header)
