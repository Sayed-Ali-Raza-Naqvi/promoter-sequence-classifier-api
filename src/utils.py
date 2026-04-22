import numpy as np


BASE_TO_INDEX = {
    'A': 0,
    'C': 1,
    'G': 2,
    'T': 3,
}
VALID_BASES = set(BASE_TO_INDEX.keys())
SEQUENCE_LENGTH = 600

def is_valid_sequence(seq: str) -> bool:
    seq = seq.upper()
    return len(seq) == SEQUENCE_LENGTH and all(base in VALID_BASES for base in seq)


def one_hot_encode(seq: str) -> np.ndarray:
    seq = seq.upper()
    L = len(seq)
    one_hot = np.zeros((4, L), dtype=np.float32)

    for i, base in enumerate(seq):
        if base in BASE_TO_INDEX:
            one_hot[BASE_TO_INDEX[base], i] = 1.0

    return one_hot


def shuffle_sequence(seq: str) -> str:
    seq_list = list(seq)
    np.random.shuffle(seq_list)
    return ''.join(seq_list)


def validate_and_encode_batch(sequences: list[str]) -> tuple[np.ndarray, list[int]]:
    encoded = []
    skipped = []

    for i, seq in sequences:
        if not is_valid_sequence(seq):
            skipped.append(i)
            continue

        encoded.append(one_hot_encode(seq))

    if not encoded:
        raise ValueError("No valid sequences found in the batch.")

    return np.stack(encoded, axis=0), skipped