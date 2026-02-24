import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class TabularDataset(Dataset):
    def __init__(self, X_np: np.ndarray, y_np: np.ndarray | None = None):
        self.X = torch.from_numpy(X_np.astype(np.float32))
        self.y = None if y_np is None else torch.from_numpy(y_np.astype(np.float32)).view(-1, 1)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if self.y is None:
            return self.X[idx]
        return self.X[idx], self.y[idx]

def make_loaders(X_train, y_train, X_val, y_val, batch_size: int):
    train_loader = DataLoader(TabularDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TabularDataset(X_val, y_val),     batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
