import torch
import torch.nn as nn


class CNNModel(nn.Module):
    def __init__(self, input_size=(512, 512)) -> None:
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Calcular el tamaño de salida del bloque de características
        conv_output_size = self._get_conv_output(input_size)

        # Bloque clasificador: capas totalmente conectadas
        self.fc1 = nn.Linear(conv_output_size, 2)

    def forward(self, x):
        x = self._forward_features(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x

    def _forward_features(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

    def _get_conv_output(self, input_size):
        # Pasamos un tensor de prueba para calcular el tamaño de salida
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, input_size[0], input_size[1])
            output_feat = self._forward_features(dummy_input)
        return output_feat.view(1, -1).size(1)