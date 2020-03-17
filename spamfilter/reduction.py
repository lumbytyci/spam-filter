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


def do_work(header: dict, difficulty: int) -> str:
    print(forge_header(header))
    h = ''

    while not h.startswith('000000'):
        pow_header['seed'] += 1
        h = hashlib.sha256(forge_header(header).encode()).hexdigest()

    print(h)


if __name__ == "__main__":
    do_work(pow_header, 5)
