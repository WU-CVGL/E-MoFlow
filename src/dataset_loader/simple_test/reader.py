import os
import torch
import logging
import numpy as np

from pathlib import Path
from ..base import DatasetBase

logger = logging.getLogger(__name__)

class SimpleTestDataset(DatasetBase):
    def __init__(self, data_path: Path, threshold: float, range: list,
                 hight: int, weight: int, color_event: bool = False) -> None:
        self.data_path = Path(data_path)
        self.threshold = threshold
        self.hight, self.weight = hight, weight
        self.color_event = color_event
        self.range_indices = range

        self.txt_files_path = self._load_txt_files_path(self.data_path, self.range_indices)
        self.event_start_time, self.event_end_time = self._get_event_duration(self.txt_files_path)
        self.duration = self.event_end_time - self.event_start_time
        logger.info(f"Event stream duration: {self.duration} sec")

    def _load_txt_files_path(self, dictionary, range_indices) -> str:
        txt_files = [f for f in os.listdir(dictionary) if f.endswith('.txt') and f[:-4].isdigit()]
        sorted_txt_files = sorted(txt_files, key=lambda x: int(x[:-4]))
        sorted_txt_files_path = [os.path.join(dictionary, f) for f in sorted_txt_files]
        start, end = range_indices
        return sorted_txt_files_path[start:end+1]
    
    def _get_event_duration(self, files) -> float:
        first_frame_events, last_frame_events = np.loadtxt(files[0]), np.loadtxt(files[-1])
        start_time, end_time = first_frame_events[0][0], last_frame_events[-1][0]
        return start_time, end_time
    
    # @property
    def data_num(self) -> int:
        return len(self.txt_files_path)

    # @property
    def name(self) -> str:
        return SimpleTestDataset.__name__
       