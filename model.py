import torch
import torch.nn as nn


class SimpleRegressionModel(nn.Module):
    def __init__(self, input_size):
        super(SimpleRegressionModel, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, 512)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(p=0.1)

        # Output 3 values: lower bound, mean, upper bound
        self.fc3 = nn.Linear(256, 3)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


class HuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta

    def forward(self, predictions, targets):
        """
        predictions: tensor of shape (batch_size, 3) - lower, mean, upper bounds
        targets: tensor of shape (batch_size, 3) - true lower, mean, upper bounds
        """
        error = predictions - targets
        abs_error = torch.abs(error)

        quadratic = 0.5 * error.pow(2)
        linear = self.delta * abs_error - 0.5 * (self.delta ** 2)

        loss = torch.where(abs_error <= self.delta, quadratic, linear)
        return loss.mean()