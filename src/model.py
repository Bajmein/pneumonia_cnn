import torch
import torch.nn as nn


class CNNModel(nn.Module):
    def __init__(self, layer_config, input_size, drop_out=None) -> None:  # noqa
        super(CNNModel, self).__init__()

        # Bloque de características: convoluciones, activaciones y pooling
        self.features = nn.Sequential(
            nn.Conv2d(1, layer_config[0], kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(layer_config[0], layer_config[1], kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(layer_config[1], layer_config[2], kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Calcular el tamaño de salida del bloque de características
        conv_output_size = self._get_conv_output(input_size)

        # Bloque clasificador: capas totalmente conectadas
        self.classifier = nn.Sequential(
            nn.Linear(conv_output_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _get_conv_output(self, input_size):
        # Pasamos un tensor de prueba para calcular el tamaño de salida
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, *input_size)
            output_feat = self.features(dummy_input)
        return output_feat.view(1, -1).size(1)