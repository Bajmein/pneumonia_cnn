from pathlib import Path
import torch
from torch.nn import CrossEntropyLoss
from src.data_processing import load_data
from src.training import train
from src.model import CNNModel
from utils.experiment_utils import create_model_filename


if __name__ == "__main__":
    try:
        # General Configuration
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        OUTPUT_PATH = Path("../outputs")
        OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

        TRAIN_PATH = "../data/split/train"
        VALIDATION_PATH = "../data/split/eval"
        TEST_PATH = "../data/split/test"

        # TODO: Externalizar configuraciones
        EPOCHS = 20
        BATCH_SIZE = 16
        LR = 0.001
        EARLY_STOPPING_PATIENCE = 5
        DROPOUT = None
        LAYER_CONFIG = [32, 64, 128]
        INPUT_SIZE = (256, 256)

        if DEVICE.type == "cuda":
            print(f"\nDevice: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        else:
            print("\nDevice: CPU")

        # Data Preparation
        train_loader, val_loader, test_loader = load_data(
            batch_size=BATCH_SIZE,
            train_path=TRAIN_PATH,
            val_path=VALIDATION_PATH,
            test_path=TEST_PATH,
        )

        # Model Initialization
        model = CNNModel(LAYER_CONFIG, INPUT_SIZE).to(DEVICE)
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
            patience=EARLY_STOPPING_PATIENCE,
            best_model_path=OUTPUT_PATH / (
                model_filename := create_model_filename(
                    model, optimizer, criterion, scheduler, DROPOUT, EPOCHS, LR, LAYER_CONFIG
                )
            )
        )

    except KeyboardInterrupt:
        print("\nTraining interrupted.")