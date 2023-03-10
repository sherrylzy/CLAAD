import torch
import torchaudio

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def freq_mask(S_dB, R=10):
    L_Fmask = int(S_dB.size()[0]) / R
    S_Fmask = torchaudio.transforms.FrequencyMasking(freq_mask_param=L_Fmask)(
        S_dB
    )
    return S_Fmask
