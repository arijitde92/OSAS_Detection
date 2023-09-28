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
        x_data = np.moveaxis(self.x[index], [1, 0], [0, 1])
        y_label = self.y[index]

        return torch.tensor(x_data, dtype=torch.float32),\
            torch.unsqueeze(torch.tensor(y_label, dtype=torch.float32), dim=0)
