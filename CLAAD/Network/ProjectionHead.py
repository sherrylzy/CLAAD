import os
from typing import Any, Callable, List, Optional, Type, Union

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import scipy.signal as signal
import seaborn as sns
import torch
import torchaudio
from IPython.display import Audio
from pytorch_metric_learning import losses
from scipy.io import wavfile
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Projection(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=512, output_dim=128):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim, bias=True),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            # nn.Linear(self.hidden_dim, self.hidden_dim, bias=True),
            # nn.BatchNorm1d(self.hidden_dim),
            # nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim, bias=False),
        )

    def forward(self, x):
        x = self.model(x)
        return F.normalize(x, dim=1)
