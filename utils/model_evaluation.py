import torch
from sklearn.metrics import accuracy_score


def evaluate_model(model, data_loader, device, return_preds=False):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calcular precisión general
    accuracy = accuracy_score(all_labels, all_preds)

    if return_preds:
        return all_labels, all_preds

    return accuracy
