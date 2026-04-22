import os
import sys
import torch
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from model import PromoterCNN
from utils import one_hot_encode, is_valid_sequence, SEQUENCE_LENGTH


MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'cnn_promoter.pt')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
THRESHOLD = 0.5

def load_model() -> PromoterCNN:
    model = PromoterCNN(dropout=0.5).to(DEVICE)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    print(f"Model loaded from {MODEL_PATH}")
    print(f"Running on {DEVICE}")
    
    return model


def predict_sequence(sequence: str, model: PromoterCNN) -> dict:
    sequence = sequence.upper().strip()

    if not is_valid_sequence(sequence):
        raise ValueError(
            f"Invalid sequence. Must be {SEQUENCE_LENGTH} characters long and contain only A, C, G, T."
            f"Got length {len(sequence)}: {sequence}"
        )
    
    encoded = one_hot_encode(sequence)
    encoded_tensor = torch.tensor(encoded, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(encoded_tensor)
        probability = torch.sigmoid(logits).item()

    label = 'promoter' if probability >= THRESHOLD else 'non-promoter'
    confidence = probability if probability >= THRESHOLD else 1 - probability

    return {
        "label": label,
        "confidence": round(confidence, 4),
        "probability": round(probability, 4),
        "length": len(sequence),
    }


def batch_predict(sequences: list[str], model: PromoterCNN) -> list[dict]:
    results = []

    for i, seq in enumerate(sequences):
        seq = seq.upper().strip()

        if not is_valid_sequence(seq):
            results.append({
                "index": i,
                "label": "error",
                "message": f"Invalid sequence at index {i}: length={len(seq)}"
            })

            continue

        result = predict_sequence(seq)
        result["index"] = i
        results.append(result)

    return results