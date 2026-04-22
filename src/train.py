import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataset import PromoterDataset
from model import PromoterCNN, count_parameters


DATA_DIR = '../data/processed/'
MODEL_PATH = '../models/'
PLOT_PATH = '../models/training_curves.png'

BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 1e-3
DROPOUT_RATE = 0.5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_loaders() -> tuple[DataLoader, DataLoader]:
    train_dataset = PromoterDataset(
        X_path=os.path.join(DATA_DIR, 'X_train.npy'),
        y_path=os.path.join(DATA_DIR, 'y_train.npy')
    )
    val_dataset = PromoterDataset(
        X_path=os.path.join(DATA_DIR, 'X_val.npy'),
        y_path=os.path.join(DATA_DIR, 'y_val.npy')
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader


def train_one_epoch(
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y_batch)
        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct += (preds == y_batch).sum().item()
        total += len(y_batch)

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


def validate(
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            total_loss += loss.item() * len(y_batch)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == y_batch).sum().item()
            total += len(y_batch)

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


def plot_curves(
    train_losses: list[float],
    val_losses: list[float],
    train_accs: list[float],
    val_accs: list[float],
    save_path: str
) -> None:
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, train_losses, label="Train Loss", color="steelblue")
    ax1.plot(epochs, val_losses, label="Val Loss", color="coral")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("BCE Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, train_accs, label="Train Acc", color="steelblue")
    ax2.plot(epochs, val_accs, label="Val Acc", color="coral")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Curves saved: {save_path}")


def main():
    print(f"Device: {DEVICE}")
    os.makedirs(MODEL_PATH, exist_ok=True)

    # Data
    train_loader, val_loader = get_loaders()

    # Model
    model = PromoterCNN(dropout=DROPOUT_RATE).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5
    )

    print(f"Model parameters: {count_parameters(model):,}")
    print(f"Training for {EPOCHS} epochs | Batch size: {BATCH_SIZE}\n")

    # Tracking
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_loss = float("inf")
    best_epoch = 0

    # ── Epoch loop ─────────────────────────────────────────────────────────────
    for epoch in range(1, EPOCHS + 1):

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate(model, val_loader, criterion)

        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(MODEL_PATH, "cnn_promoter.pt"))
            print(f" ✓ Best model saved (epoch {epoch})")

    # ── Post training ──────────────────────────────────────────────────────────
    print(f"\nTraining complete.")
    print(f"Best model: Epoch {best_epoch} | Val Loss: {best_val_loss:.4f}")

    plot_curves(train_losses, val_losses, train_accs, val_accs, PLOT_PATH)

    # ── Test set evaluation ────────────────────────────────────────────────────
    print("\nEvaluating on test set...")

    test_ds = PromoterDataset(
        X_path=os.path.join(DATA_DIR, "X_test.npy"),
        y_path=os.path.join(DATA_DIR, "y_test.npy")
    )
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Load best weights for test evaluation
    model.load_state_dict(torch.load(os.path.join(MODEL_PATH, "cnn_promoter.pt"),
                                     map_location=DEVICE))
    test_loss, test_acc = validate(model, test_loader, criterion)
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()