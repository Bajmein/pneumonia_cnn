import torch
from colorama import Fore, Style
from utils.model_evaluation import evaluate_model


def train(model, train_loader, val_loader, test_loader, device, epochs, optimizer, criterion, model_path, scheduler=None):
    # TODO: Probar tqdm en los epochs
    # TODO: Intentar usar metodo para refrescar terminal

    # Inicializar listas para almacenar métricas por época
    train_accuracies, val_accuracies = [], []
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        # Training loop
        train_loss, train_accuracy = run_epoch(
            model, train_loader, device, optimizer, criterion, training=True
        )

        # Validation loop
        val_loss, val_accuracy = run_epoch(
            model, val_loader, device, optimizer, criterion, training=False
        )

        # Almacenar métricas
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f"\n{Fore.LIGHTWHITE_EX}Epoch {epoch + 1}/{epochs}{Style.RESET_ALL}")
        print(f"- Training:    Loss -> {Fore.GREEN}{train_loss:.4f}{Style.RESET_ALL},\
         Accuracy -> {Fore.GREEN}{train_accuracy:.4f}{Style.RESET_ALL}")
        print(f"- Validation:  Loss -> {Fore.GREEN}{val_loss:.4f}{Style.RESET_ALL},\
         Accuracy -> {Fore.GREEN}{val_accuracy:.4f}{Style.RESET_ALL}")

        if scheduler: scheduler.step(val_loss)

    # Evaluar precisión general del modelo actual
    current_evaluation_accuracy = evaluate_model(model, test_loader, device)
    print(f"\nCurrent model evaluation accuracy: {current_evaluation_accuracy:.4f}")

    # Comparar con el modelo previo guardado
    try:
        checkpoint = torch.load(model_path, weights_only=True)
        previous_evaluation_accuracy = checkpoint.get("evaluation_accuracy", 0.0)
        print(f"\nPrevious model evaluation accuracy: {previous_evaluation_accuracy:.4f}")

    except FileNotFoundError:
        previous_evaluation_accuracy = 0.0
        print("\nNo previous model found. Saving the current model as the best.")

    # Guardar el modelo si la precisión actual es mejor
    if current_evaluation_accuracy > previous_evaluation_accuracy:
        torch.save(
            {"state_dict": model.state_dict(), "evaluation_accuracy": current_evaluation_accuracy}, model_path
        )
        print(f"\nNew best model saved with evaluation accuracy: {current_evaluation_accuracy:.4f}\
         at {model_path}")
    else:
        print("\nCurrent model did not outperform the previous model.")


def run_epoch(model, loader, device, optimizer, criterion, training):
    model.train() if training else model.eval()
    total_loss, correct_predictions = 0, 0

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            # Reset gradients if training
            if training: optimizer.zero_grad()

            # Forward pass
            predictions = model(images)
            loss = criterion(predictions, labels)

            # Backward pass and optimization if training
            if training:
                loss.backward()
                optimizer.step()

            # Accumulate loss and accuracy
            total_loss += loss.item()
            correct_predictions += (predictions.argmax(dim=1) == labels).float().sum().item()  # noqa

    # Calculate average metrics
    average_loss = total_loss / len(loader.dataset)
    average_accuracy = correct_predictions / len(loader.dataset)
    return average_loss, average_accuracy
