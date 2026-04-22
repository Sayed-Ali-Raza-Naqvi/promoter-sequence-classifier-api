import torch
import torch.nn as nn


class PromoterCNN(nn.Module):
    def __init__(self, dropout=0.5):
        super(PromoterCNN, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=32, kernel_size=11, padding=5),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.global_pool = nn.AdaptiveAvgPool1d(output_size=1)

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        x = self.global_pool(x)
        x = x.squeeze(-1)
        x = self.classifier(x)
        
        return x.squeeze(-1)
    

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)