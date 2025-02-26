import os
import torch
import logging
import numpy as np

from pathlib import Path
from src.utils import misc
from typing import List, Union
from ..base import DatasetBase

logger = logging.getLogger(__name__)

class SyntheticFigureDataset(DatasetBase):

    NAME = "SyntheticFigure"

    def __init__(
        self,
        data_path: Path, 
        threshold: float, 
        batch_range: int,
        hight: int, weight: int, 
        sequence_indices: Union[List[int],int],
        color_event: bool = False
    ) -> None:
        self.data_path = Path(data_path)
        self.threshold = threshold
        self.hight, self.weight = hight, weight
        self.color_event = color_event
        self.sequence_indices = sequence_indices
        self.batch_range = batch_range

        self.event_txt_dir = self.data_path / "events"
        self.flow_txt_dir = self.data_path / "forward_flow"
        self.intrinsic_path = self.data_path / "K_matrix.txt"
        self.pose_path = self.data_path / "camera_pose.txt"

        self.events_txt_files_path = self._load_events_txt_files_path(self.event_txt_dir)
        self.train_event_files_path = misc.get_filenames(self.events_txt_files_path, sequence_indices)
        self.event_start_time, self.event_end_time = self._get_event_duration(self.train_event_files_path)
        self.duration = self.event_end_time - self.event_start_time
        logger.info(f"Event stream duration: {self.duration} sec")

        self.K = self._load_camera_intrinsic(self.intrinsic_path)
        self.gt_pose = self._load_gt_camera_pose(self.pose_path)

    def _get_event_duration(
        self, 
        files: List[str]
    ) -> float:
        if len(files) == 1:
            frame_events = np.loadtxt(files[0])
            start_time, end_time = frame_events[0][0], frame_events[-1][0]
        else:
            first_frame_events, last_frame_events = np.loadtxt(files[0]), np.loadtxt(files[-1])
            start_time, end_time = first_frame_events[0][0], last_frame_events[-1][0]
        return start_time, end_time
    
    def _load_events_txt_files_path(
        self, 
        dir: str, 
    ) -> List[str]:
        txt_files = [f for f in os.listdir(dir) if f.endswith('.txt') and f[:-4].isdigit()]
        sorted_txt_files = sorted(txt_files, key=lambda x: int(x[:-4]))
        sorted_txt_files_path = [os.path.join(dir, f) for f in sorted_txt_files]
        return sorted_txt_files_path
    
    def _load_camera_intrinsic(
        self,
        file_path: str
    ):
        K_tensor = misc.load_camera_intrinsic(file_path)
        return K_tensor

    def _load_gt_camera_pose(
        self,
        file_path: str,
    )-> List[str]:
        pose_tensor = misc.load_camera_pose(file_path)
        return pose_tensor
    
    @property
    def data_num(self) -> int:
        return len(self.train_event_files_path)

    @property
    def name(self) -> str:
        return SyntheticFigureDataset.NAME