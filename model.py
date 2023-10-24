import torch
import torch.nn as nn
import torch.nn.functional as torch_func
from torchinfo import summary

class ConvNet(nn.Module):
    def __init__(self, input_shape, dropout=0.2, num_classes=2) -> None:
        super(ConvNet, self).__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.cnn = nn.Sequential(
            # nn.Conv1d(in_channels=self.input_shape[1], out_channels=32, kernel_size=5, stride=2, padding='same',
                      # bias=False),
            nn.Conv1d(in_channels=self.input_shape[1], out_channels=32, kernel_size=5, padding='same',
                      bias=False),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3),
            # nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding='same', bias=False),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding='same', bias=False),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3),
            nn.Dropout(dropout)
        )
        if self.num_classes == 2:
            self.num_classes = 1
        self.fcn = nn.Sequential(
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=self.num_classes)
        )

    def forward(self, x):
        print("Input Shape:", x.size())
        x = self.cnn(x)
        print("CNN output shape:")
        print(x.shape)
        if torch.isnan(x).any():
            print("NaN found after CNN")
        x = x.reshape(x.size(0), -1)
        print("Shape after flattening:")
        print(x.shape)
        x = self.fcn(x)
        print("Shape after FCN:")
        print(x.shape)
        if torch.isnan(x).any():
            print("NaN found after FCN")
        if self.num_classes == 1:
            return torch_func.sigmoid(x)
        else:
            return torch_func.log_softmax(x, dim=-1)


if __name__ == '__main__':
    input_size = (256, 3, 1)    # Batch size = 256
    model = ConvNet((10000, 3))
    summary(model, input_size)
