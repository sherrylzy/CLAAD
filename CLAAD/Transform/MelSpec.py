import os
from typing import Any, Callable, List, Optional, Type, Union

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as signal
import seaborn as sns
import torch
import torchaudio
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset


def MelSpec(y, sr):
    S = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=2048,
        hop_length=512,
        f_max=sr / 2,
        norm="slaney",
        mel_scale="slaney",
    )(y)
    S_dB = torchaudio.transforms.AmplitudeToDB()(S)
    return S_dB
