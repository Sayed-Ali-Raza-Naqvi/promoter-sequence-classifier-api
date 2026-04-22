import os
import sys
import torch
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from model import PromoterCNN
from utils import one_hot_encode, normalize_sequence, SEQUENCE_LENGTH


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
    sequence = sequence.replace('\n', '').replace('\r', '').replace(' ', '').upper().strip()
    original_length = len(sequence)

    normalized_seq, warning = normalize_sequence(sequence)
    
    encoded = one_hot_encode(normalized_seq)
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
        "original_length": original_length,
        "processed_length": SEQUENCE_LENGTH,
        "warning": warning
    }


def batch_predict(sequences: list[str], model: PromoterCNN) -> list[dict]:
    results = []

    for i, seq in enumerate(sequences):
        try:
            result = predict_sequence(seq, model)
            result["index"] = i
            results.append(result)
        except ValueError as e:
            results.append({
                "index": i,
                "label": "error",
                "message": str(e)
            })

    return results