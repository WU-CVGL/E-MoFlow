import os
import torch
import logging
import numpy as np

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

    def get_batch_event_data(self, idx) -> dict:
        event_file_path = self.dataset.txt_files_path[idx]
        events = torch.from_numpy(np.loadtxt(event_file_path))

        mask = (0 <= events[:, 1]) & (events[:, 1] < self.W) & \
                (0 <= events[:, 2]) & (events[:, 2] < self.H)
        events = events[mask]
        events_norm_t = events.clone()
        events_norm_t[:, 0] = (events[:, 0] - self.t_start) / (self.t_end - self.t_start)

        batch_data : dict[str, torch.Tensor] = {
            "events": events_norm_t,
            "timestamps": events[0][:],
            "t_start": torch.Tensor([self.t_start]),
            "t_end": torch.Tensor([self.t_end])
        }

        return batch_data
    
    def collate_batch_data(self):
        pass

    def __len__(self) -> int:
        return self.dataset.data_num

    def __getitem__(self, index) -> dict:
       return self.get_batch_event_data(index)