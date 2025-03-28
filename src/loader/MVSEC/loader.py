import os
import logging

import h5py
import numpy as np

from typing import Tuple
from .base import DataLoaderBase
from ...utils import event_proc, flow_proc

logger = logging.getLogger(__name__)

# hdf5 data loader
def h5py_loader(path: str):
    """Basic loader for .hdf5 files.
    Args:
        path (str) ... Path to the .hdf5 file.

    Returns:
        timestamp (dict) ... Doctionary of numpy arrays. Keys are "left" / "right".
        davis_left (dict) ... "event": np.ndarray.
        davis_right (dict) ... "event": np.ndarray.
    """
    data = h5py.File(path, "r")
    event_timestamp = get_timestamp_index(data)
    r = {
        "event": np.array(data["davis"]["right"]["events"], dtype=np.int16),
    }
    # 'gray_ts': np.array(data['davis']['right']['image_raw_ts'], dtype=np.float64)
    l = {
        "event": np.array(data["davis"]["left"]["events"], dtype=np.int16),
        "gray_ts": np.array(data["davis"]["left"]["image_raw_ts"], dtype=np.float64),
    }
    data.close()
    return event_timestamp, l, r


def get_timestamp_index(h5py_data):
    """Timestampm loader for pre-fetching before actual sensor data loading.
    This is necessary for sync between sensors and decide which timestamp to
    be used as ground clock.
    Args:
        h5py_data... h5py object.
    Returns:
        timestamp (dict) ... Doctionary of numpy arrays. Keys are "left" / "right".
    """
    timestamp = {}
    timestamp["right"] = np.array(h5py_data["davis"]["right"]["events"][:, 2])
    timestamp["left"] = np.array(h5py_data["davis"]["left"]["events"][:, 2])
    return timestamp

class MVSECDataLoader(DataLoaderBase):
    """Dataloader class for MVSEC dataset."""

    NAME = "MVSEC"

    def __init__(self, config: dict = {}):
        super().__init__(config)

    def get_events_for_train(self, t1=None, t2=None):
        self.t1_train = self.min_ts
        self.t2_train = self.max_ts
        
        if t1 is not None:
            self.t1_train = t1
        if t2 is not None:
            self.t2_train = t2
            
        if self.t1_train >= self.t2_train:
            logger.error(f"Invalid time: t1({self.t1_train}) >= t2({self.t2_train})")
            raise ValueError(f"Invalid time: t1({self.t1_train}) >= t2({self.t2_train})")

        ind1 = self.time_to_index(self.t1_train)  # event index
        ind2 = self.time_to_index(self.t2_train)
        events_for_train = self.load_event(ind1, ind2)
        return events_for_train
         
    def get_events_for_valid(self, t1=10, t2=15):
        self.t1_valid = self.min_ts + t1
        self.t2_valid = self.min_ts + t2
            
        if self.t1_valid >= self.t2_valid:
            logger.error(f"Invalid time: t1({self.t1_valid}) >= t2({self.t2_valid})")
            raise ValueError(f"Invalid time: t1({self.t1_valid}) >= t2({self.t2_valid})")
        ind1 = self.time_to_index(self.t1_valid)  # event index
        ind2 = self.time_to_index(self.t2_valid)
        events_for_valid = self.load_event(ind1, ind2)
        return events_for_valid
    
    # Override
    def set_sequence(self, sequence_name: str, undistort: bool = False) -> None:
        logger.info(f"Use sequence {sequence_name}")
        self.sequence_name = sequence_name
        logger.info(f"Undistort events = {undistort}")

        self.dataset_files = self.get_sequence(sequence_name)
        ts, l_event, r_event = h5py_loader(self.dataset_files["event"])
        self.left_event = l_event["event"]  # int16 .. for smaller memory consumption.
        self.left_ts = ts["left"]  # float64
        self.left_gray_ts = l_event["gray_ts"]  # float64
        
        logger.info(f"Loading {len(self.left_ts)} events from {sequence_name} in total")
        logger.info(f"{sequence_name} comprises {len(self.left_gray_ts)} images in total")

        # self.right_event = r_event["event"]
        # self.right_ts = ts["right"]
        # self.right_gray_ts = r_event["gray_ts"]  # float64

        # Setup gt and filter
        if self.gt_flow_available:
            self.setup_gt_flow(os.path.join(self.gt_flow_dir, sequence_name))
            self.setup_gt_motion(os.path.join(self.gt_flow_dir, sequence_name))
            self.omit_invalid_data(sequence_name)

        # Calib param
        calib_param = self.load_calib(sequence_name)
        self.intrinsic = calib_param["K"]
        self.distortion_coeffs = calib_param["D"]
        
        # Undistort - most likely necessary to run evaluation with GT.
        self.undistort = undistort
        if self.undistort:
            self.calib_map_x, self.calib_map_y = self.get_calib_map(
                self.dataset_files["calib_map_x"], self.dataset_files["calib_map_y"]
            )

        # Setting up time suration statistics
        self.min_gray_ts = self.left_gray_ts.min()
        self.max_gray_ts = self.left_gray_ts.max()
        self.min_ts = self.left_ts.min()
        self.max_ts = self.left_ts.max()
        # self.min_ts = np.max([self.left_ts.min(), self.right_ts.min()])
        # self.max_ts = np.min([self.left_ts.max(), self.right_ts.max()]) - 10.0  # not use last 1 sec
        self.data_duration = self.max_ts - self.min_ts
        logger.info(f"The time length of {sequence_name} is {self.data_duration}s")
        
    def get_sequence(self, sequence_name: str) -> dict:
        """Get data inside a sequence.

        Inputs:
            sequence_name (str) ... name of the sequence. ex) `outdoot_day2`.

        Returns:
            sequence_file (dict) ... dictionary of the filenames for the sequence.
        """
        data_path: str = os.path.join(self.dataset_dir, sequence_name)
        event_file = data_path + "_data.hdf5"
        
        self.calib_dir: str = os.path.join(self.dataset_dir, sequence_name[:-1] + "_calib")
        calib_path: str = os.path.join(self.calib_dir, sequence_name[:-1])
        calib_file_x = calib_path + "_left_x_map.txt"
        calib_file_y = calib_path + "_left_y_map.txt"
        
        sequence_file = {
            "event": event_file,
            "calib_map_x": calib_file_x,
            "calib_map_y": calib_file_y,
        }
        return sequence_file

    def setup_gt_flow(self, path):
        path = path + "_gt_flow_dist.npz"
        gt = np.load(path)
        self.gt_timestamps = gt["timestamps"]
        self.U_gt_all = gt["x_flow_dist"]
        self.V_gt_all = gt["y_flow_dist"]
        logger.info(f"Loading ground truth flow {path}")
        logger.info(f"{self.sequence_name} provides GT flow at {len(self.gt_timestamps)} timestamps")
        
    def setup_gt_motion(self, path):
        path = path + "_odom.npz"
        gt_motion = np.load(path)
        self.gt_motion_timestamps = gt_motion["timestamps"]
        self.lin_vel_gt_all = gt_motion["lin_vel"]
        self.ang_vel_gt_all = gt_motion["ang_vel"]
        logger.info(f"Loading ground truth motion {path}")
        logger.info(f"{self.sequence_name} provides GT motion at {len(self.gt_motion_timestamps)} timestamps")

    def free_up_flow(self):
        del self.gt_timestamps, self.U_gt_all, self.V_gt_all

    def omit_invalid_data(self, sequence_name: str):
        logger.info(f"Use only valid frames.")
        first_valid_gt_frame = 0
        last_valid_gt_frame = -1
        if "indoor_flying1" in sequence_name:
            first_valid_gt_frame = 60
            last_valid_gt_frame = 1340
        elif "indoor_flying2" in sequence_name:
            first_valid_gt_frame = 140
            last_valid_gt_frame = 1500
        elif "indoor_flying3" in sequence_name:
            first_valid_gt_frame = 100
            last_valid_gt_frame = 1711
        elif "indoor_flying4" in sequence_name:
            first_valid_gt_frame = 104
            last_valid_gt_frame = 380
        elif "outdoor_day1" in sequence_name:
            last_valid_gt_frame = 5020
        elif "outdoor_day2" in sequence_name:
            first_valid_gt_frame = 30
            # last_valid_gt_frame = 5020
            
        # print(np.array_equal(self.gt_timestamps, self.gt_motion_timestamps))
        self.gt_timestamps = self.gt_timestamps[first_valid_gt_frame:last_valid_gt_frame]
        self.U_gt_all = self.U_gt_all[first_valid_gt_frame:last_valid_gt_frame]
        self.V_gt_all = self.V_gt_all[first_valid_gt_frame:last_valid_gt_frame]
        self.lin_vel_gt_all = self.lin_vel_gt_all[first_valid_gt_frame:last_valid_gt_frame]
        self.ang_vel_gt_all = self.ang_vel_gt_all[first_valid_gt_frame:last_valid_gt_frame]

        # Update event list
        first_event_index = self.time_to_index(self.gt_timestamps[0])
        last_event_index = self.time_to_index(self.gt_timestamps[-1])
        self.left_event = self.left_event[first_event_index:last_event_index]
        self.left_ts = self.left_ts[first_event_index:last_event_index]

        self.min_ts = self.left_ts.min()
        self.max_ts = self.left_ts.max()

        # Update gray frame ts
        self.left_gray_ts = self.left_gray_ts[
            (self.gt_timestamps[0] < self.left_gray_ts)
            & (self.gt_timestamps[-1] > self.left_gray_ts)
        ]
        self.left_gray_ts_dt = np.diff(self.left_gray_ts)
        
        logger.info(f"Filter and obtain {len(self.left_ts)} valid events.")
        logger.info(f"Filter and obtain {len(self.left_gray_ts)} valid images.")
        logger.info(f"Filter and obtain {len(self.gt_timestamps)} valid ground truth flow.")
        logger.info(f"Filter and obtain {len(self.gt_timestamps)} valid ground truth motion.")
        logger.info(f"The average frame interval of grayscale images is {np.mean(self.left_gray_ts_dt)}.")

        # self.right_event = self.right_event[first_event_index:last_event_index]
        # self.right_ts = self.right_ts[first_event_index:last_event_index]
        # self.right_gray_ts = self.right_gray_ts[
        #     (self.gt_timestamps[0] < self.right_gray_ts)
        #     & (self.gt_timestamps[-1] > self.right_gray_ts)
        # ]

    def __len__(self):
        return len(self.left_event)

    def load_event(self, start_index: int, end_index: int, cam: str = "left") -> np.ndarray:
        """Load events.
        The original hdf5 file contains (x, y, t, p),
        where x means in width direction, and y means in height direction. p is -1 or 1.

        Returns:
            events (np.ndarray) ... Events. [x, y, t, p] where x is height.
            t is absolute value, in sec. p is [-1, 1].
        """
        n_events = end_index - start_index + 1
        events = np.zeros((n_events, 4), dtype=np.float64)

        if cam == "left":
            if len(self.left_event) <= start_index:
                logger.error(
                    f"Specified {start_index} to {end_index} index for {len(self.left_event)}."
                )
                raise IndexError
            events[:, 0] = self.left_event[start_index:end_index+1, 1]
            events[:, 1] = self.left_event[start_index:end_index+1, 0]
            events[:, 2] = self.left_ts[start_index:end_index+1]
            events[:, 3] = self.left_event[start_index:end_index+1, 3]
        elif cam == "right":
            logger.error("Please select `left`as `cam` parameter.")
            raise NotImplementedError
        if self.undistort:
            events = event_proc.undistort_events(
                events, self.calib_map_x, self.calib_map_y, self._HEIGHT, self._WIDTH
            )
        return events

    # Optical flow (GT)
    def gt_time_list(self):
        return self.gt_timestamps

    def eval_frame_time_list(self):
        # In MVSEC, evaluation is based on gray frame timestamp.
        return self.left_gray_ts

    def index_to_time(self, index: int) -> float:
        return self.left_ts[index]

    def time_to_index(self, time: float) -> int:
        if time < self.left_ts.min() or time > self.left_ts.max():
            logger.error("The time is out of range.")
            raise ValueError("The time is out of range.")
        
        ind = np.searchsorted(self.left_ts, time)
        
        if ind < len(self.left_ts) and self.left_ts[ind] == time:
            return ind 
        else:
            return ind - 1  

    def get_gt_time(self, index: int) -> tuple:
        """Get GT flow timestamp [floor, ceil] for a given index.

        Args:
            index (int): Index of the event

        Returns:
            tuple: [floor_gt, ceil_gt]. Both are synced with GT optical flow.
        """
        inds = np.where(self.gt_timestamps > self.index_to_time(index))[0]
        if len(inds) == 0:
            return (self.gt_timestamps[-1], None)
        elif len(inds) == len(self.gt_timestamps):
            return (None, self.gt_timestamps[0])
        else:
            return (self.gt_timestamps[inds[0] - 1], self.gt_timestamps[inds[0]])

    def load_optical_flow(self, t1: float, t2: float) -> np.ndarray:
        """Load GT Optical flow based on timestamp.
        Note: this is pixel displacement.
        Note: the args are not indices, but timestamps.

        Args:
            t1 (float): [description]
            t2 (float): [description]

        Returns:
            [np.ndarray]: H x W x 2. Be careful that the 2 ch is [height, width] direction component.
        """
        U_gt, V_gt = flow_proc.estimate_corresponding_gt_flow(
            self.U_gt_all,
            self.V_gt_all,
            self.gt_timestamps,
            t1,
            t2,
        )
        gt_flow = np.stack((V_gt, U_gt), axis=2)
        return gt_flow

    def load_gt_motion(self):
        timestamps_col = self.gt_timestamps.reshape(-1, 1) - self.min_ts
        self.lin_vel_gt_all = np.hstack([timestamps_col, self.lin_vel_gt_all])  # shape (1280, 4)
        self.ang_vel_gt_all = np.hstack([timestamps_col, self.ang_vel_gt_all])  # shape (1280, 4)
        return self.lin_vel_gt_all, self.ang_vel_gt_all
    
    def load_calib(self, sequence_name) -> dict:
        """Load calibration file.

        Outputs:
            (dict) ... {"K": camera_matrix, "D": distortion_coeff}
                camera_matrix (np.ndarray) ... [3 x 3] matrix.
                distortion_coeff (np.array) ... [4] array.
        """
        logger.warning("directly load calib_param is not implemented!! please use rectify instead.")
        intrinsics_mat, distortion_coeffs = [], []
        
        if(sequence_name[:-1] == "indoor_flying"):
            intrinsics_mat = np.array(
                [
                    [226.0181418548734, 0, 174.5433576736815],
                    [0, 225.7869434267677, 124.21627572590607],
                    [0, 0, 1],
                ],
                dtype=np.float32,
            )
            distortion_coeffs = np.array(
                [
                    -0.04846669832871334,
                    0.010092844338123635,
                    -0.04293073765014637,
                    0.005194706897326005,
                ],
                dtype=np.float32,
            )
        elif(sequence_name[:-1] == "outdoor_day"):
            intrinsics_mat = np.array(
                [
                    [223.9940010790056, 0, 170.7684322973841],
                    [0, 223.61783486959376, 128.18711828338436],
                    [0, 0, 1],
                ],
                dtype=np.float32,
            )
            distortion_coeffs = np.array(
                [
                    -0.033904378348448685,
                    -0.01537260902537579,
                    -0.022284741346941413,
                    0.0069204143687187645,
                ],
                dtype=np.float32,
            )
        else:
            logger.error(f"{sequence_name} is not exist.")
            raise ValueError
        return {"K": intrinsics_mat, "D": distortion_coeffs}

    def get_calib_map(self, map_txt_x, map_txt_y):
        """Intrinsic calibration parameter file loader.
        Args:
            map_txt... file path.
        Returns
            map_array (np.array)... map array.
        """
        map_x = self.load_map_txt(map_txt_x)
        map_y = self.load_map_txt(map_txt_y)
        return map_x, map_y

    def load_map_txt(self, map_txt):
        f = open(map_txt, "r")
        line = f.readlines()
        map_array = np.zeros((self._HEIGHT, self._WIDTH))
        for i, l in enumerate(line):
            map_array[i] = np.array([float(k) for k in l.split()])
        return map_array
