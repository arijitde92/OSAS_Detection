import torch
import torch.nn as nn
import torch.nn.functional as torch_func
from torchinfo import summary


class ConvNet(nn.Module):
    def __init__(self, input_shape, dropout=0.5, num_classes=2) -> None:
        super(ConvNet, self).__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=self.input_shape[1], out_channels=32, kernel_size=5, stride=2),
            # nn.Conv1d(in_channels=self.input_shape[1], out_channels=32, kernel_size=5, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=2),
            # nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3),
            nn.Dropout(dropout)
        )
        # if self.num_classes == 2:
        #     self.num_classes = 1
        self.fcn = nn.Sequential(
            nn.Linear(in_features=1024, out_features=32),
            # nn.Linear(in_features=1536, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=self.num_classes)
        )

    def forward(self, x):
        # print("Input Shape:", x.size())
        x = self.cnn(x)
        # print("CNN output shape:")
        # print(x.shape)
        if torch.isnan(x).any():
            print("NaN found after CNN")
        x = x.reshape(x.size(0), -1)
        # print("Shape after flattening:")
        # print(x.shape)
        x = self.fcn(x)
        # print("Shape after FCN:")
        # print(x.shape)
        if torch.isnan(x).any():
            print("NaN found after FCN")
        # if self.num_classes == 1:
        #     return torch_func.log_softmax(x, dim=-1)
        # else:
        #     return torch_func.log_softmax(x, dim=-1)
        return x


class ConvNetMultiHead(nn.Module):
    def __init__(self, input_shape, dropout=0.5, num_classes=2) -> None:
        super(ConvNetMultiHead, self).__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.cnn1 = nn.Sequential(
            nn.Conv1d(in_channels=self.input_shape[1], out_channels=32, kernel_size=5, stride=2),
            # nn.Conv1d(in_channels=self.input_shape[1], out_channels=32, kernel_size=5, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=2),
            # nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3),
            nn.Dropout(dropout)
        )

        self.cnn2 = nn.Sequential(
            nn.Conv1d(in_channels=self.input_shape[1], out_channels=32, kernel_size=7, stride=2),
            # nn.Conv1d(in_channels=self.input_shape[1], out_channels=32, kernel_size=5, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=7, stride=2),
            # nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3),
            nn.Dropout(dropout)
        )

        self.cnn3 = nn.Sequential(
            nn.Conv1d(in_channels=self.input_shape[1], out_channels=32, kernel_size=11, stride=2),
            # nn.Conv1d(in_channels=self.input_shape[1], out_channels=32, kernel_size=5, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=11, stride=2),
            # nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3),
            nn.Dropout(dropout)
        )
        # if self.num_classes == 2:
        #     self.num_classes = 1
        self.fcn = nn.Sequential(
            nn.Linear(in_features=2880, out_features=720),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=720, out_features=180),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=180, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=self.num_classes)
        )

    def forward(self, x):
        # print("Input Shape:", x.size())
        x1 = self.cnn1(x)
        x2 = self.cnn2(x)
        x3 = self.cnn3(x)
        # print("CNN output shape:")
        # print(x.shape)
        if torch.isnan(x).any():
            print("NaN found after CNN")
        x1 = x1.reshape(x1.size(0), -1)
        x2 = x2.reshape(x2.size(0), -1)
        x3 = x3.reshape(x3.size(0), -1)
        # print("Shape after flattening:")
        # print("X1: ", x1.shape)
        # print("X2: ", x2.shape)
        # print("X3: ", x3.shape)

        conc_x = torch.concatenate((x1, x2, x3), dim=1)
        # print("Concatenated Shape:", conc_x.shape)
        x = self.fcn(conc_x)
        # print("Shape after FCN:")
        # print(x.shape)
        if torch.isnan(x).any():
            print("NaN found after FCN")
        # if self.num_classes == 1:
        #     return torch_func.log_softmax(x, dim=-1)
        # else:
        #     return torch_func.log_softmax(x, dim=-1)
        return x


if __name__ == '__main__':
    input_size = (256, 5, 600)  # Batch size = 256
    model1 = ConvNet((10000, 5, 600), num_classes=5)
    model2 = ConvNetMultiHead((10000, 5, 600), num_classes=5)
    summary(model2, input_size)
