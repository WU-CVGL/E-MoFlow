import os
import abc
import logging
import numpy as np

logger = logging.getLogger(__name__)

class DatasetBase(object):
    def __init__(self):
        pass
    
    @property
    @abc.abstractmethod
    def name(self):
        raise NotImplementedError
    @property
    @abc.abstractmethod
    def data_num(self):
        raise NotImplementedError

    # def set_sequence(self, sequence_name: str) -> None:
    #     logger.info(f"Use sequence {sequence_name}")
    #     self.sequence_name = sequence_name
    #     self.dataset_files = self.get_sequence(sequence_name)

    # def get_sequence(self, sequence_name: str) -> dict:
    #     raise NotImplementedError

    # def load_event(
    #     self, start_index: int, end_index: int, cam: str = "left", *args, **kwargs
    # ) -> np.ndarray:
    #     raise NotImplementedError

    # def load_calib(self) -> dict:
    #     raise NotImplementedError

    # def load_optical_flow(self, t1: float, t2: float, *args, **kwargs) -> np.ndarray:
    #     raise NotImplementedError

    # def index_to_time(self, index: int) -> float:
    #     raise NotImplementedError

    # def time_to_index(self, time: float) -> int:
    #     raise NotImplementedError
