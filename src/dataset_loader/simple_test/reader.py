import os
import torch
import logging
import numpy as np

from pathlib import Path

logger = logging.getLogger(__name__)

class SimpleTest():
    def __init__(self, data_path: Path, threshold: float, 
                 hight: int, weight: int, color_event: bool = False) -> None:
        self.data_path = data_path
        self.threshold = threshold
        self.hight, self.weight = hight, weight
        self.color_event = color_event
        # logger.info(f"dictionary name: {self.data_path}")
        self.event_txt_files = self.load_event_txt_files(self.data_path)

    def load_event_txt_files(self, dictionary):
        txt_files = [f for f in os.listdir(dictionary) if f.endswith('.txt') and f[:-4].isdigit()]
        sorted_txt_files = sorted(txt_files, key=lambda x: int(x[:-4]))
        for text_file in sorted_txt_files:
            logger.info(f"files name: {text_file}")
        return sorted_txt_files
       