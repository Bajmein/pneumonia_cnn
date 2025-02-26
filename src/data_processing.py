import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from pathlib import Path
from PIL import Image
import numpy as np


class ImageDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx, 0]
        label = int(self.dataframe.iloc[idx, 1])

        img = Image.open(img_path).convert('L')

        img = self.transform(img) if self.transform else torch.tensor(np.array(img), dtype=torch.float32).unsqueeze(0) / 255.0
        return img, torch.tensor(label, dtype=torch.long)


def create_dataframe(folder_path):
    folder_path = Path(folder_path)
    image_list = []

    for class_name in ['normal', 'pneumonia']:
        class_path = folder_path / class_name
        if not class_path.exists():
            raise FileNotFoundError(f"The path {class_path} does not exist.")
        for file in class_path.iterdir():
            image_list.append([str(file), class_name])

    return pd.DataFrame(image_list, columns=['file_path', 'label'])


def label_dataframe(df):
    label_mapping = {'normal': 0, 'pneumonia': 1}
    df['label'] = df['label'].map(label_mapping)
    return df

def calculate_dataset_mean_std(dataset, batch_size=16):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    mean = 0.0
    std = 0.0
    total_samples = 0

    for images, _ in loader:
        batch_samples = images.size(0)  # Batch size
        images = images.view(batch_samples, images.size(1), -1)  # Flatten the image pixels
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_samples += batch_samples

    mean /= total_samples
    std /= total_samples

    return mean.numpy(), std.numpy()

def load_data(batch_size, train_path, val_path, test_path):
    # Create DataFrames
    train_df = label_dataframe(create_dataframe(train_path))
    val_df = label_dataframe(create_dataframe(val_path))
    test_df = label_dataframe(create_dataframe(test_path))

    # with open("../outputs/dataset_stats.json", "r") as f:
    #     config = json.load(f)
    #
    # mean = config.get("mean")
    # std = config.get("std")

    # Transformations
    train_transform = transforms.Compose([
        # TODO: Evaluar agregar transformaciones
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    # Create datasets
    # TODO: Usar una red GAN para balancear la clase de entrenamiento
    train_dataset = ImageDataset(train_df, transform=train_transform)
    val_dataset = ImageDataset(val_df, transform=val_test_transform)
    test_dataset = ImageDataset(test_df, transform=val_test_transform)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
