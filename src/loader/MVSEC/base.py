import os
import logging
import numpy as np

from ...utils import misc

logger = logging.getLogger(__name__)

class DataLoaderBase(object):
    """Base of the DataLoader class.
    Please make sure to implement
     - load_event()
     - get_sequence()
    in chile classes.
    """

    NAME = "example"

    def __init__(self, config: dict = {}):
        self._HEIGHT = config["hight"]
        self._WIDTH = config["width"]

        self.dataset_name = config["dataset_name"]
        root_dir: str = config["data_path"] 
        self.root_dir: str = os.path.expanduser(root_dir)
        self.dataset_dir: str = os.path.join(self.root_dir, config["sequence"][:-1])
        logger.info(f"Loading data in {self.dataset_dir}")
        
        self.__dataset_files: dict = {}
        
        self.gt_flow_available: bool
        if misc.check_key_and_bool(config, "load_gt_flow"):
            gt_flow_dir: str = os.path.expanduser(config["gt_path"])
            self.gt_flow_dir: str = os.path.join(gt_flow_dir, config["sequence"][:-1])
            self.gt_flow_available = misc.check_file_utils(self.gt_flow_dir)
            logger.info(f"Loading ground truth in {self.gt_flow_dir}")
        else:
            self.gt_flow_available = False

        if misc.check_key_and_bool(config, "undistort"):
            logger.info("Undistort events when load_event.")
            self.auto_undistort = True
        else:
            logger.info("No undistortion.")
            self.auto_undistort = False

    @property
    def dataset_files(self) -> dict:
        return self.__dataset_files

    @dataset_files.setter
    def dataset_files(self, sequence: dict):
        self.__dataset_files = sequence

    def set_sequence(self, sequence_name: str) -> None:
        logger.info(f"Use sequence {sequence_name}")
        self.sequence_name = sequence_name
        self.dataset_files = self.get_sequence(sequence_name)

    def get_sequence(self, sequence_name: str) -> dict:
        raise NotImplementedError

    def load_event(
        self, start_index: int, end_index: int, cam: str = "left", *args, **kwargs
    ) -> np.ndarray:
        raise NotImplementedError

    def load_calib(self) -> dict:
        raise NotImplementedError

    def load_optical_flow(self, t1: float, t2: float, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError
    
    def load_gt_motion(self) -> np.ndarray:
        raise NotImplementedError

    def index_to_time(self, index: int) -> float:
        raise NotImplementedError

    def time_to_index(self, time: float) -> int:
        raise NotImplementedError
