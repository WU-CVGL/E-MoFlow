import os
import torch
import logging
import numpy as np

from src.utils import misc
from typing import Dict, Any, Union, List
from torch.utils.data import Dataset

from .reader import SyntheticFigureDataset

logger = logging.getLogger(__name__)

class SyntheticFigureDataProvider(Dataset):
    def __init__(self, dataset: SyntheticFigureDataset) -> None:
        super().__init__()
        self.dataset = dataset
        self.H = dataset.hight
        self.W = dataset.weight
        self.t_start = dataset.event_start_time
        self.t_end = dataset.event_end_time
        self.duration = dataset.duration
        self.batch_range = dataset.batch_range

    def get_train_data(self, idx) -> Dict:
        event_file_paths = self.dataset.train_event_files_path
        event_data_list = [np.loadtxt(file) for file in event_file_paths]
        events = torch.from_numpy(np.vstack(event_data_list))

        events_norm = misc.process_events(
            origin_events=events,
            image_size=(self.H, self.W),
            start_end=(self.t_start,self.t_end)
        )

        train_batch_data : Dict[str, Any] = {
            "events": events,
            "events_norm": events_norm,
            "timestamps": events[:, 0],
        }

        return train_batch_data
    
    def get_valid_data(
        self, 
        valid_event_file_indices: Union[List[int],int]
    ) -> torch.Tensor:
        valid_event_file_paths = misc.get_filenames(
            self.dataset.txt_files_path, valid_event_file_indices
        )
        valid_event_data_list = [np.loadtxt(file) for file in valid_event_file_paths]
        valid_events = torch.from_numpy(np.vstack(valid_event_data_list))

        valid_events_norm = misc.process_events(
            origin_events=valid_events,
            image_size=(self.H, self.W),
            start_end=(self.t_start,self.t_end)
        )

        valid_batch_data : Dict[str, Any] = {
            "events": valid_events,
            "events_norm": valid_events_norm,
            "timestamps": valid_events[:, 0],
        }

        return valid_batch_data
        
    def collate_batch_data(self):
        pass

    # def __len__(self) -> int:
    #     return self.dataset.data_num
    
    def __len__(self) -> int:
        return 1

    def __getitem__(self, index) -> Dict:
       return self.get_train_data(index)