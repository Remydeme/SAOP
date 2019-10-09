

#

from __future__ import print_function

import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader, TensorDataset



class LassoRegression(nn.Module):

    lr = 1e-3

    def __init__(self, input_size, output):
        super(LassoRegression, self).__init__()
        self.theta = nn.Linear(in_features=input_size, out_features=output, bias=True)


    def forward(self, x):
        """ Compute the value of the output """
        output = self.theta(x)
        return output



class Lasso:
    lr = 1e-3
    def __init__(self, batch_size=25, epochs=1000):
        self.bacth_size = batch_size
        self.epochs = epochs
        self.criterion = nn.MSELoss()


    def fit(self, x, y):
        X = torch.from_numpy(x).float()
        Y = torch.from_numpy(y).float()
        data = TensorDataset(X, Y)
        self.dataset = DataLoader(data, batch_size=self.bacth_size)
        self.regressor = LassoRegression(input_size=x.shape[1],output=y.shape[1])


    def transform(self):
        """ Compute the gradient"""

        for epoch in range(1, self.epochs):
            for xb, yb in self.dataset:
                y_hat = self.regressor(xb)
                error = self.criterion(y_hat, yb)

                # compute the gradient
                error.backward()

                with torch.no_grad():

                    for p in self.regressor.parameters():
                        p -= self.lr * p.grad

                    self.regressor.zero_grad()

        for p in self.regressor.parameters():
             print("Theta parameter :{}".format(p))





if __name__ == "__main__":
    reg = Lasso()
    x = np.random.randn(100, 4)
    y =  x @ np.array([[2],[3],[4],[5]]) + 23
    reg.fit(x=x, y=y)
    reg.transform()