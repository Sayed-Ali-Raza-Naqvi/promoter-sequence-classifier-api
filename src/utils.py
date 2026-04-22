import numpy as np


BASE_TO_INDEX = {
    'A': 0,
    'C': 1,
    'G': 2,
    'T': 3,
}
VALID_BASES = set(BASE_TO_INDEX.keys())
SEQUENCE_LENGTH = 600
MIN_LENGTH = 500

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


def pad_sequence(seq: str, target_length: int = SEQUENCE_LENGTH) -> str:
    seq = seq.upper().strip()
    total_pad = target_length - len(seq)
    left_pad = total_pad // 2
    right_pad = total_pad - left_pad

    return 'N' * left_pad + seq + 'N' * right_pad


def trim_to_center(seq: str, target_length: int = SEQUENCE_LENGTH) -> str:
    seq = seq.upper().strip()
    start = (len(seq) - target_length) // 2
    
    return seq[start:start + target_length]


def normalize_sequence(seq: str,) -> tuple[str, str | None]:
    seq = seq.upper().strip()
    length = len(seq)
    warning = None

    invalid_bases = set(seq) - VALID_BASES

    if invalid_bases:
        raise ValueError(f"Sequence contains invalid bases: {invalid_bases}")
    
    if length == SEQUENCE_LENGTH:
        pass
    elif length > SEQUENCE_LENGTH:
        seq = trim_to_center(seq)
        warning = (
            f"Sequence was {length} bp — trimmed to center "
            f"{SEQUENCE_LENGTH} bp window. "
            f"Prediction is reliable."
        )
    elif length >= MIN_LENGTH:
        seq = pad_sequence(seq)
        warning = (
            f"Sequence was {length} bp — padded to center "
            f"{SEQUENCE_LENGTH} bp window. "
            f"Prediction is less reliable."
        )
    else:
        raise ValueError(
            f"Sequence too short: {length} bp. "
            f"Minimum accepted length is {MIN_LENGTH} bp. "
            f"Model was trained on {SEQUENCE_LENGTH} bp sequences. "
            f"Heavy padding below {MIN_LENGTH} bp produces unreliable predictions."
        )

    return seq, warning


def shuffle_sequence(seq: str) -> str:
    seq_list = list(seq)
    np.random.shuffle(seq_list)
    return ''.join(seq_list)


def validate_and_encode_batch(sequences: list[str]) -> tuple[np.ndarray, list[int]]:
    encoded = []
    skipped = []

    for i, seq in enumerate(sequences):
        if not is_valid_sequence(seq):
            skipped.append(i)
            continue

        encoded.append(one_hot_encode(seq))

    if not encoded:
        raise ValueError("No valid sequences found in the batch.")

    return np.stack(encoded, axis=0), skipped