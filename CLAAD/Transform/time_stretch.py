import librosa
import librosa.display
import numpy as np
import torch
import torchaudio
from torch import Tensor, nn
from torch.nn import functional as F

from Transform.MelSpec import MelSpec


def time_stretch(y, R_min=0.5, R_max=2, sr=16000):
    r = np.random.uniform(R_min, R_max)
    y_stretch = librosa.effects.time_stretch(y.detach().numpy(), rate=r)
    y_stretch = librosa.resample(y_stretch, sr, sr * R)
    y_stretch = torch.from_numpy(y_stretch)
    S = MelSpec(y_stretch, sr)
    return S
