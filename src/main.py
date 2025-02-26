from pathlib import Path
import torch
from torch.nn import CrossEntropyLoss
from src.data_processing import load_data
from src.training import train
from src.model import CNNModel


if __name__ == "__main__":
    try:
        # General Configuration
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nDevice: {torch.cuda.get_device_name()}") if DEVICE.type == "cuda" else print("\nDevice: CPU")

        OUTPUT_PATH = Path("../outputs")
        OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

        TRAIN_PATH = "../data/divided/train"
        VALIDATION_PATH = "../data/divided/eval"
        TEST_PATH = "../data/divided/test"

        # TODO: Externalizar configuraciones
        EPOCHS = 20
        BATCH_SIZE = 32
        LR = 0.0001

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
        scheduler = None

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
            best_model_path=OUTPUT_PATH / 'CustomCNN.pth',
        )

    except KeyboardInterrupt:
        print("\nTraining interrupted.")