import numpy as np
import torch
from torch.utils.data import Dataset


class PromoterDataset(Dataset):
    def __init__(self, X_path, y_path):
        self.X = np.load(X_path)
        self.y = np.load(y_path)

        assert len(self.X) == len(self.y), "The number of samples in X and y must be the same."

        print(f"Loaded dataset with {len(self.X)} samples. |"
              f"X shape: {self.X.shape}, y shape: {self.y.shape} |"
              f"Promoters: {int(self.y.sum())} |"
              f"Background: {int((1 - self.y).sum())}" 
        )

    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.float32)

        return x, y