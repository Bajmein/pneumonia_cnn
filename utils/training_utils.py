import torch
import torch.nn as nn


class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_val_loss = float("inf")
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            print(f"\nNo improvement in val_loss. Patience counter: {self.counter}/{self.patience}")

        if self.counter >= self.patience:
            self.early_stop = True

        return self.early_stop


def calculate_class_weights(data_loader, num_classes):
    class_counts = torch.zeros(num_classes)

    # Contar ocurrencias de cada clase
    for _, labels in data_loader:
        for label in labels:
            class_counts[label] += 1

    # Calcular pesos inversos normalizados
    class_weights = 1.0 / class_counts
    normalized_weights = class_weights / class_weights.sum()

    return normalized_weights


class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = torch.nn.functional.cross_entropy(inputs, targets, reduction="none", weight=self.alpha)
        p_t = torch.exp(-ce_loss)  # Probabilidad predicha
        focal_loss = (1 - p_t) ** self.gamma * ce_loss
        return focal_loss.mean()
