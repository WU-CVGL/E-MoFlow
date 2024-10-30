import os
import torch
import logging
import numpy as np

from typing import Dict, Any
from torch.utils.data import Dataset

from .reader import SimpleTestDataset

logger = logging.getLogger(__name__)

class SimpleTestDataProvider(Dataset):
    def __init__(self, dataset: SimpleTestDataset) -> None:
        super().__init__()
        self.dataset = dataset
        self.H = dataset.hight
        self.W = dataset.weight
        self.t_start = dataset.event_start_time
        self.t_end = dataset.event_end_time
        self.duration = dataset.duration
        self.batch_range = dataset.batch_range

    def get_batch_event_data(self, idx) -> Dict:
        event_file_paths = self.dataset.txt_files_path[idx:idx+self.batch_range]
        event_data_list = [np.loadtxt(file) for file in event_file_paths]
        events = torch.from_numpy(np.vstack(event_data_list))

        mask = (0 <= events[:, 1]) & (events[:, 1] < self.W) & \
                (0 <= events[:, 2]) & (events[:, 2] < self.H)
        events = events[mask]
        events_sorted, sort_indices = torch.sort(events[:, 0])
        events_sorted = events[sort_indices]

        events_norm = events_sorted.clone()
        events_norm[:, 0] = (events[:, 0] - self.t_start) / (self.t_end - self.t_start) 
        events_norm[:, 1] = events[:, 1] / self.W
        events_norm[:, 2] = events[:, 2] / self.H

        batch_data : Dict[str, Any] = {
            "events": events_sorted,
            "events_norm": events_norm,
            "timestamps": events[:, 0],
        }

        return batch_data
    
    def collate_batch_data(self):
        pass

    def __len__(self) -> int:
        return self.dataset.data_num

    def __getitem__(self, index) -> Dict:
       return self.get_batch_event_data(index)