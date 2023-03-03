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

from Dataset.data_loader import MIMII, audiodir

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_test(train_size, device="valve", id=0):
    train_size = 0.75
    dir, label = audiodir(device, id)
    dir_abnormal, label_abnormal = audiodir(device, id, Data="abnormal")

    dataset_normal = MIMII(dir, label)
    dataset_abnormal = MIMII(dir_abnormal, label_abnormal)
    train_dataset, test_normal_dataset = torch.utils.data.random_split(
        dataset_normal,
        [
            int(len(dataset_normal) * train_size),
            len(dataset_normal) - int(len(dataset_normal) * train_size),
        ],
    )

    test_dataset = ConcatDataset([test_normal_dataset, dataset_abnormal])

    Train = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )
    Test = torch.utils.data.DataLoader(
        test_dataset, batch_size=300, shuffle=True, num_workers=4
    )
    return Train, Test