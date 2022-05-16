import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F
import os
import pandas as pd
from torchvision.io import read_image
import numpy as np
import pandas as pd



device = "cpu"
if torch.cuda.is_available():
     device = "cuda"

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = nn.Linear(15,500)
        self.l2 = nn.Linear(500,500)
        self.l3 = nn.Linear(500,1)
    def forward(self, x):
        x = torch.tanh(self.l1(x))
        x = torch.tanh(self.l2(x))
        x = self.l3(x)
        return x
    def perform_training(self, dataloader, epochs=100):
        loss_history = []
        optim = torch.optim.Adam(self.parameters(), 0.001)
        for t in range(epochs):
            print(f"Epoch {t + 1}\n-------------------------------")
            size = len(dataloader.dataset)
            for batch, (X, y) in enumerate(dataloader):
                # Compute prediction and loss
                pred = self(X)
                loss = F.mse_loss(pred, y.unsqueeze(1))
                # Backpropagation
                optim.zero_grad()
                loss.backward()
                optim.step()
                if batch % 100 == 0:
                    loss, current = loss.item(), batch * len(X)
                    print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
                loss_history.append(loss)
        return loss_history

if __name__ == "__main__":
    d = pd.read_csv("walmart_cleaned.csv")
    d = d.drop("Date", axis=1)
    d = d.astype("float32")
    d = d.apply(lambda x: ((x - x.mean())/x.std()))
    label = d["Weekly_Sales"]
    data = d.drop("Weekly_Sales", axis=1)
    data_tensor = torch.tensor(data.values).to(device)
    label_tensor = torch.tensor(label.values).to(device)
    dataset = TensorDataset(data_tensor, label_tensor)
    train_loader = DataLoader(dataset, batch_size=1000, shuffle=True, drop_last=False,  )
    n = Net().to(device)
    n.perform_training(train_loader)