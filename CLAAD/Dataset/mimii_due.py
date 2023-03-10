import glob
import os

import librosa
import numpy as np
import scipy
from torch.utils.data import ConcatDataset, Dataset
from tqdm import tqdm




class MIMII_DUE(Dataset):
    """Parameters:
    -----------
    target_dir : str
        base directory path
    section_name : str
        section name of audio file in <<dir_name>> directory
    dir_name : str
        sub directory name (train or source_test or target_test)
    mode : str
        'development' - [default] (with leabels normal/anomal)
        'evaluation' - audio without lables
    return:
    --------
        if the mode is "development":
            data: audio_files_features according extraction_type: numpy.array (numpy.array( float )) vector array
            labels : np.array [ boolean ]
                label info. list
                * normal/anomaly = 0/1
        if the mode is "evaluation":
             data: audio_files_features, np.array (numpy.array( float )) vector array
    """

    def __init__(
            self,
            target_dir: str,
            section_name: str,
            dir_name: str,
            mode: str , #mode: str = "development"
    ):
        self.target_dir = target_dir
        self.section_name = section_name
        self.dir_name = dir_name
        self.mode = mode

        self.samples = []
        self.file_list, self.labels = self._init_file_list_generator()
        self.data = []
        self._file_list_to_data()

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> tuple:
        return self.data[idx], self.labels[idx]


    def _mimii_due_file_list_generator(self):
        if self.mode == "development":
            # development
            query = os.path.abspath(
                f"{self.target_dir}/{self.dir_name}/{self.section_name}_*_normal_*.wav"
            )
            normal_files = sorted(glob.glob(query))
            normal_labels = np.zeros(len(normal_files))

            query = os.path.abspath(
                "{target_dir}/{dir_name}/{section_name}_*_anomaly_*.wav".format(
                    target_dir=self.target_dir,
                    dir_name=self.dir_name,
                    section_name=self.section_name,
                )
            )
            anomaly_files = sorted(glob.glob(query))
            anomaly_labels = np.ones(len(anomaly_files))

            file_list = np.concatenate((normal_files, anomaly_files), axis=0)
            labels = np.concatenate((normal_labels, anomaly_labels), axis=0)
        else:
            # evaluation
            query = os.path.abspath(
                "{target_dir}/{dir_name}/{section_name}_*.wav".format(
                    target_dir=self.target_dir,
                    dir_name=self.dir_name,
                    section_name=self.section_name,
                )
            )
            file_list = [sorted(glob.glob(query))]
            labels = [None]

        return file_list, labels

