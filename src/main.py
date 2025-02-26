from pathlib import Path
import random
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from src.data_processing import load_data
from src.training import train
from src.model import CNNModel


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # Para múltiples GPUs

if __name__ == "__main__":
    try:
        # Fijar semilla aleatoria
        set_seed(42)

        # General Configuration
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nDevice: {torch.cuda.get_device_name()}") if DEVICE.type == "cuda" else print("\nDevice: CPU")

        OUTPUT_PATH = Path("../outputs")
        OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

        TRAIN_PATH = "../data/divided/train"
        VALIDATION_PATH = "../data/divided/val"
        TEST_PATH = "../data/divided/test"

        # TODO: Externalizar configuraciones
        EPOCHS = 10
        BATCH_SIZE = 64
        LR = 3e-4

        # Data Preparation
        train_loader, val_loader, test_loader = load_data(
            batch_size=BATCH_SIZE,
            train_path=TRAIN_PATH,
            val_path=VALIDATION_PATH,
            test_path=TEST_PATH,
        )

        # Model Initialization
        model = CNNModel().to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = CrossEntropyLoss()

        # Training
        train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=DEVICE,
            epochs=EPOCHS,
            optimizer=optimizer,
            criterion=criterion,
            model_path=OUTPUT_PATH / 'BasicCNN.pth',
        )

    except KeyboardInterrupt:
        print("\nTraining interrupted.")