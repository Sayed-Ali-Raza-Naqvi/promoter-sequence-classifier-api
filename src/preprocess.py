import os
import numpy as np
from Bio import SeqIO
from sklearn.model_selection import train_test_split
from utils import is_valid_sequence, one_hot_encode, shuffle_sequence,validate_and_encode_batch, SEQUENCE_LENGTH
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_PROMOTER_DIR = BASE_DIR / "data" / "raw" / "hg38_edyfo.fa"
OUTPUT_DIR = BASE_DIR / "data" / "processed"

RANDOM_SEED = 42
VAL_SIZE = 0.10
TEST_SIZE = 0.10

def load_promoter_sequences(fasta_path: str) -> list[str]:
    sequences = []
    total = 0
    skipped = 0

    for record in SeqIO.parse(fasta_path, "fasta"):
        total += 1
        seq = str(record.seq).upper()

        if len(seq) > SEQUENCE_LENGTH:
            start = (len(seq) - SEQUENCE_LENGTH) // 2
            seq = seq[start:start + SEQUENCE_LENGTH]

        if not is_valid_sequence(seq):
            skipped += 1
            continue

        sequences.append(seq)

    print(f"Promoters -- Total: {total} | Valid: {len(sequences)} | Skipped: {skipped}")
    return sequences


def generate_background_sequences(promoter_seqs: list[str]) -> list[str]:
    np.random.seed(RANDOM_SEED)
    background_seqs = []

    for seq in promoter_seqs:
        bg_seq = shuffle_sequence(seq)
        background_seqs.append(bg_seq)
    
    print(f"Background sequences generated: {len(background_seqs)}")
    return background_seqs


def encode_and_label(promoters: list[str], background: list[str]) -> tuple[np.ndarray, np.ndarray]:
    all_sequences = promoters + background
    all_labels = [1] * len(promoters) + [0] * len(background)

    X = np.stack([one_hot_encode(seq) for seq in all_sequences], axis=0)
    y = np.array(all_labels, dtype=np.float32)

    print(f"Encoded sequences: {X.shape}, Labels: {y.shape}")
    print(f"Class distribution -- Promoters: {np.sum(y==1)} | Background: {np.sum(y==0)}")

    return X, y


def split_data(X: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_SEED
    )

    relative_val_size = VAL_SIZE / (1.0 - TEST_SIZE)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=relative_val_size,
        stratify=y_train_val,
        random_state=RANDOM_SEED
    )

    splits = {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test
    }

    for name, data in splits.items():
        print(f"{name}: {data.shape}")

    return splits


def save_splits(splits: dict[str, np.ndarray], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    for name, data in splits.items():
        path = os.path.join(output_dir, f"{name}.npy")
        np.save(path, data)
        print(f"Saved {name} to {path}")


def sanity_check(output_dir: str) -> None:
    print("\n── Sanity Check ──")

    for split in ["train", "val", "test"]:
        X = np.load(os.path.join(output_dir, f"X_{split}.npy"))
        y = np.load(os.path.join(output_dir, f"y_{split}.npy"))

        print(f"{split} raw shape: {X.shape}")

        assert len(X) == len(y), f"Mismatch in X and y lengths for {split}"

        if X.ndim != 3:
            raise AssertionError(f"{split} is not 3D!")

        if X.shape[-1] == 4:
            row_sums = X.sum(axis=2)
        elif X.shape[1] == 4:
            row_sums = X.sum(axis=1)
        else:
            raise AssertionError(f"Invalid shape for X_{split}: {X.shape}")

        if not np.allclose(row_sums, 1.0):
            bad_positions = np.where(~np.isclose(row_sums, 1.0))
            print(f"Found invalid one-hot entries in {split} set!")
            raise AssertionError(f"One-hot encoding broken in {split} set!")

        assert np.all((X >= 0) & (X <= 1)), f"Invalid values in X_{split}"
        assert set(np.unique(y)).issubset({0, 1}), f"Invalid labels in y_{split}"

        print(f"{split:5s} — X: {X.shape} | y: {y.shape} | "
              f"Promoters: {int(y.sum())} | "
              f"Background: {len(y) - int(y.sum())} | "
              f"One-hot OK: True")
        

if __name__ == "__main__":
    print("=" * 50)
    print("STEP 1: Loading promoter sequences")
    print("=" * 50)
    promoters = load_promoter_sequences(RAW_PROMOTER_DIR)

    print("\n" + "=" * 50)
    print("STEP 2: Generating background sequences")
    print("=" * 50)
    background = generate_background_sequences(promoters)

    print("\n" + "=" * 50)
    print("STEP 3: Encoding + labeling")
    print("=" * 50)
    X, y = encode_and_label(promoters, background)

    print("\n" + "=" * 50)
    print("STEP 4: Splitting data")
    print("=" * 50)
    splits = split_data(X, y)

    print("\n" + "=" * 50)
    print("STEP 5: Saving arrays")
    print("=" * 50)
    save_splits(splits, OUTPUT_DIR)

    print("\n" + "=" * 50)
    print("STEP 6: Sanity check")
    print("=" * 50)
    sanity_check(OUTPUT_DIR)