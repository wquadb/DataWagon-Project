import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

from tqdm import tqdm
import argparse
import random
import json
import os


device = torch.device("cpu")
print("\nStarting with device:", device)


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed) cuda seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}\n")


class NeuralNetwork(nn.Module):
    def __init__(self, in_features: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hidden_features   = 36
        hidden_features_2 = 18

        self.linear_layer = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.LeakyReLU(0.01),
            nn.BatchNorm1d(hidden_features),
            nn.Dropout(p=0.25),

            nn.Linear(hidden_features, hidden_features_2),
            nn.LeakyReLU(0.01),
            nn.Dropout(p=0.1),

            nn.Linear(hidden_features_2, 1)
        )

    def forward(self, x):
        y_pred = F.sigmoid(self.linear_layer(x))
        return y_pred

