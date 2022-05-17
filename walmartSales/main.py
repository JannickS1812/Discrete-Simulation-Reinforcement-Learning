import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

device = "cpu"
if torch.cuda.is_available():
     device = "cuda"
print(f'Device: {device}')

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = nn.Linear(14,500)
        self.l2 = nn.Linear(500,500)
        self.l3 = nn.Linear(500,1)

    def forward(self, x):
        x = torch.tanh(self.l1(x))
        x = torch.tanh(self.l2(x))
        return self.l3(x)

    def train(self,
              train_loader,
              val_loader=None,
              epochs=100,
              validation_fails=10):

        train_loss_history = []
        val_loss_history = []
        val_fail_history = []
        running_val_fails = 0

        optim = torch.optim.Adam(self.parameters(), 0.001)
        num_samples = len(train_loader.dataset)
        for t in range(epochs):
            print(f"---------- Epoch {t + 1} ----------")
            for batch, (X, Y) in enumerate(train_loader):

                # Compute prediction and loss
                pred = self(X)
                loss = F.mse_loss(pred, Y.unsqueeze(1))

                # Backpropagation
                optim.zero_grad()
                loss.backward()
                optim.step()
                if batch % 100 == 0:
                    print(f"loss: {loss.item():>4f} [{batch * len(X):>5d}/{num_samples:>5d}]")

            # calculate validation error once per epoch
            train_loss_history.append([t, loss.item()])
            if val_loader is not None:
                loss_history = []
                for (X_val, Y_val) in val_loader:
                    pred_val = self(X_val)
                    loss_val = F.mse_loss(pred_val, Y_val.unsqueeze(1))
                    loss_history.append(loss_val.item())

                loss_val = np.mean(loss_history)
                val_loss_history.append([t, loss_val])
                print(f'training loss: {loss.item():>4f}, validation loss: {loss_val.item():>4f}')

                # early stopping
                if validation_fails is not None and t > 1:
                    if val_loss_history[-2][1] < val_loss_history[-1][1]: # check if validation loss increased
                        running_val_fails += 1
                    else:
                        running_val_fails = 0
                    val_fail_history.append([t, running_val_fails])
                    if running_val_fails >= validation_fails:
                        print(f'Validation loss did not decrease for {running_val_fails} epochs, training is aborted')
                        break

            else:
                print(f"loss: {loss.item():>4f} [{batch * len(X):>5d}/{num_samples:>5d}]")

        if validation_fails is None:
            return np.array(train_loss_history), np.array(val_loss_history)
        else:
            return np.array(train_loss_history), np.array(val_loss_history), np.array(val_fail_history)

if __name__ == "__main__":

    # load csv
    d = pd.read_csv("walmart_cleaned.csv")
    d = d.drop(["Date", "Unnamed: 0"], axis=1)
    d = d.astype("float32")
    Y = d["Weekly_Sales"].values
    X = d.drop("Weekly_Sales", axis=1).values

    # train-val-test split with 70-15-15
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)
    X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, test_size=0.5, random_state=1)

    # input normalization
    mean_x = X_train.mean(axis=0)
    std_x = X_train.std(axis=0)
    X_train = (X_train - mean_x) / std_x
    X_val = (X_val - mean_x) / std_x
    X_test = (X_test - mean_x) / std_x

    # output normalization
    mean_y = Y_train.mean(axis=0)
    std_y = Y_train.std(axis=0)
    Y_train = (Y_train - mean_y) / std_y
    Y_val = (Y_val - mean_y) / std_y
    Y_test = (Y_test - mean_y) / std_y

    train_loader = DataLoader(TensorDataset(torch.tensor(X_train).to(device), torch.tensor(Y_train).to(device)),
                              batch_size=1000, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val).to(device), torch.tensor(Y_val).to(device)),
                            batch_size=X_val.shape[0])

    n = Net().to(device)
    train_history, val_history, val_fail_history = n.train(train_loader, val_loader, epochs=300)

    # console output
    train_loss = train_history[-1, 1]
    val_loss = val_history[-1, 1]
    test_loss = F.mse_loss(n(torch.tensor(X_test).to(device)), torch.tensor(Y_test).to(device).unsqueeze(1)).item()
    print('\nLosses:')
    print(f'Train:      {train_loss:.4f}')
    print(f'Validation: {val_loss:.4f}')
    print(f'Test:       {test_loss:.4f}')

    # plot loss over epochs
    plt.plot(train_history[:, 0], train_history[:, 1], label='Training')
    plt.plot(val_history[:, 0], val_history[:, 1], label='Validation')
    plt.plot(val_fail_history[:, 0], val_fail_history[:, 1], label='Validation Fails')
    plt.legend()
    plt.show()


