import os
import torch
import numpy as np

from torch.utils.data import Dataset

from ..event_data import EventData
from .reader import SimpleTestDataset



class SimpleTestDataProvider(Dataset):
    def __init__(self, dataset: SimpleTestDataset, device: torch.device) -> None:
        super().__init__()
        self.dataset = dataset
        self.H = dataset.hight
        self.W = dataset.weight
        self.device = device

    def get_batch_event_data(self, idx) -> EventData:
        event_file_path = self.dataset.txt_files_path[idx]
        batch_events = EventData(np.loadtxt(event_file_path), self.H, self.W, self.device)
        return batch_events.to_tensor()

    def __len__(self) -> int:
        return len(self.dataset.data_num)

    def __getitem__(self, index) -> EventData:
       return self.get_batch_event_data(index)