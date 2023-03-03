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


class LinCLS(nn.Module):
    def __init__(self, input_dim=512, output_dim=8):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim

        self.fc1 = nn.Linear(self.input_dim, self.output_dim)

    def forward(self, x):
        x = self.fc1(x)
        return x
