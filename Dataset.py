import numpy as np
from torch.utils.data import Dataset
import torch


class OSASUDDataset(Dataset):
    def __init__(self, data: np.ndarray, labels: np.ndarray) -> None:
        super(OSASUDDataset, self).__init__()
        self.x = data
        self.y = labels

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x_data = np.reshape(self.x[index], (-1, self.x[index].shape[-1]))
        x_data = np.moveaxis(x_data, [1, 0], [0, 1])  # For PSG and ECG
        y_label = self.y[index]

        return torch.tensor(x_data, dtype=torch.float32), torch.tensor(y_label)


class ApneaECGDataset(Dataset):
    def __init__(self, data: np.ndarray, labels: np.ndarray) -> None:
        super(ApneaECGDataset, self).__init__()
        self.x = data
        self.y = labels

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        # x_data = np.reshape(self.x[index], (-1, self.x[index].shape[-1]))
        x_data = np.moveaxis(self.x[index], [1, 0], [0, 1])  # For PSG and ECG
        y_label = self.y[index]

        return torch.tensor(x_data, dtype=torch.float32), torch.from_numpy(y_label)
