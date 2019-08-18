import torch
from torch.utils.data import Dataset


class NumpyDataset(Dataset):
    def __init__(self, data, target, transform=None):
        self.data = torch.from_numpy(data / 255).float()
        self.target = torch.from_numpy(target).long()
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]

        if self.transform:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.data)
