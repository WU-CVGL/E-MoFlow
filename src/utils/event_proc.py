import math
import h5py
import torch
import numba
import weakref
import numpy as np

from pathlib import Path
from typing import Union, Dict, Tuple

def crop_event(
    events: torch.Tensor, 
    x0: int, x1: int, 
    y0: int, y1: int
) -> torch.Tensor:
    """Crop events.

    Args:
        events (Tensor): [n x 4]. [x, y, t, p].
        x0 (int): Start of the crop, at row[0]
        x1 (int): End of the crop, at row[0]
        y0 (int): Start of the crop, at row[1]
        y1 (int): End of the crop, at row[1]

    Returns:
        Tensor: Cropped events.
    """
    mask = (
        (x0 <= events[:, 0]) & (events[:, 0] < x1) &
        (y0 <= events[:, 1]) & (events[:, 1] < y1)
    )

    cropped = events[mask]
    return cropped

def undistort_events(events, map_x, map_y, h, w):
    """Undistort (rectify) events.
    Args:
        events ... [x, y, t, p]. X is height direction.
        map_x, map_y... meshgrid

    Returns:
        events... events that is in the camera plane after undistortion.
    TODO check overflow
    """
    # k = np.int32(map_y[np.int16(events[:, 1]), np.int16(events[:, 0])])
    # l = np.int32(map_x[np.int16(events[:, 1]), np.int16(events[:, 0])])
    # k = np.int32(map_y[events[:, 1].astype(np.int32), events[:, 0].astype(np.int32)])
    # l = np.int32(map_x[events[:, 1].astype(np.int32), events[:, 0].astype(np.int32)])
    # undistort_events = np.copy(events)
    # undistort_events[:, 0] = l
    # undistort_events[:, 1] = k
    # return undistort_events[((0 <= k) & (k < h)) & ((0 <= l) & (l < w))]

    k = np.int32(map_y[events[:, 0].astype(np.int32), events[:, 1].astype(np.int32)])
    l = np.int32(map_x[events[:, 0].astype(np.int32), events[:, 1].astype(np.int32)])
    undistort_events = np.copy(events)
    undistort_events[:, 0] = k
    undistort_events[:, 1] = l
    return undistort_events[((0 <= k) & (k < h)) & ((0 <= l) & (l < w))]

def normalized_plane_to_pixel(
    events: torch.Tensor,
    intrinsic_mat: torch.Tensor
) -> torch.Tensor:
    """
    Convert events coordinates from normalized camera plane to pixel coordinates.
    
    Args:
        events (torch.Tensor): Event data tensor of shape (B, N, 4) with columns (x, y, t, p)
                             where x,y are in normalized camera plane, B is batch size
        intrinsic_mat (torch.Tensor): Camera intrinsic matrix of shape (B, 3, 3) or (3, 3)
    
    Returns:
        torch.Tensor: Converted events of shape (B, N, 4) with columns (x, y, t, p)
                     where x,y are in pixel coordinates
    """
    # Handle single intrinsic matrix case
    if intrinsic_mat.dim() == 2:
        intrinsic_mat = intrinsic_mat.unsqueeze(0).expand(events.size(0), -1, -1)
    
    # Extract camera intrinsics
    fx = intrinsic_mat[:, 0, 0].unsqueeze(1)  # shape: (B, 1)
    fy = intrinsic_mat[:, 1, 1].unsqueeze(1)  # shape: (B, 1)
    cx = intrinsic_mat[:, 0, 2].unsqueeze(1)  # shape: (B, 1)
    cy = intrinsic_mat[:, 1, 2].unsqueeze(1)  # shape: (B, 1)
    
    # Extract normalized plane coordinates
    x_norm = events[:, :, 0]  # shape: (B, N)
    y_norm = events[:, :, 1]  # shape: (B, N)
    
    # Convert to pixel coordinates
    # For pinhole camera model: u = fx * x + cx, v = fy * y + cy
    x_pixel = fx * x_norm + cx  # Broadcasting: (B, 1) * (B, N) + (B, 1) -> (B, N)
    y_pixel = fy * y_norm + cy
    
    # Create output tensor keeping timestamp and polarity unchanged
    output_events = torch.stack([x_pixel, y_pixel, events[:, :, 2], events[:, :, 3]], dim=2)
    
    return output_events

def normalize_events(
    origin_events: torch.Tensor,
    image_size: Tuple[int, int],
    intrinsic_mat: torch.Tensor,
    start_end: Tuple[float,  float],
    normalize_time: bool = True,
    normalize_coords_mode: str = "NORM_PLANE"
) -> torch.Tensor:
    """
    Load and process event data files, converting them into normalized tensor vectors.
    
    Args:
        events (torch.Tensor): original event data
        image_size (Tuple[int, int]): Target image size as (height, width)
        start_end Tuple[float, float]: the start and end timestamps of whole sequence
        normalize_coords (bool): Whether to normalize spatial coordinates
        normalize_time (bool): Whether to normalize timestamps
    
    Returns:
        torch.Tensor: Processed event data as a tensor with shape (N, 4) where N is 
                   the total number of valid events and columns are (t, x, y, p)
    """

    # Mask and sort
    mask = (0 <= origin_events[:, 1]) & (origin_events[:, 1] < image_size[1]) & \
            (0 <= origin_events[:, 2]) & (origin_events[:, 2] < image_size[0])
    events = origin_events[mask]
    events_sorted, sort_indices = torch.sort(events[:, 0])
    events_sorted = events[sort_indices]

    # Extract components
    timestamps = events_sorted[:, 0]
    x_coords = events_sorted[:, 1]
    y_coords = events_sorted[:, 2]
    polarities = events_sorted[:, 3]
    
    # Normalize timestamps if requested
    if normalize_time:
        norm_timestamps = (timestamps - start_end[0]) / (start_end[1] - start_end[0])
    else:
        normalize_time = timestamps
    
    # Normalize coordinates if requested
    if normalize_coords_mode == "UV_SPACE":
        norm_x_coords = x_coords / (image_size[1] - 1)  # Width
        norm_y_coords = y_coords / (image_size[0] - 1)  # Height
    elif normalize_coords_mode == "NORM_PLANE":
        fx, fy = intrinsic_mat[0,0], intrinsic_mat[1,1]
        cx, cy = intrinsic_mat[0,2], intrinsic_mat[1,2]
        norm_x_coords = (x_coords - cx) / fx
        norm_y_coords = (y_coords - cy) / fy
    else:
        norm_x_coords = x_coords
        norm_y_coords = y_coords
    
    # Combine back into tensor
    processed_events = torch.stack([norm_timestamps, norm_x_coords, norm_y_coords, polarities], axis=1)
    return processed_events

class EventSlicer:
    def __init__(self, h5f: h5py.File):
        self.h5f = h5f
        self.events = dict()
        for dset_str in ['p', 'x', 'y', 't']:
            self.events[dset_str] = self.h5f['events/{}'.format(dset_str)]

        # This is the mapping from milliseconds to event index:
        # It is defined such that
        # (1) t[ms_to_idx[ms]] >= ms*1000, for ms > 0
        # (2) t[ms_to_idx[ms] - 1] < ms*1000, for ms > 0
        # (3) ms_to_idx[0] == 0
        # , where 'ms' is the time in milliseconds and 't' the event timestamps in microseconds.
        #
        # As an example, given 't' and 'ms':
        # t:    0     500    2100    5000    5000    7100    7200    7200    8100    9000
        # ms:   0       1       2       3       4       5       6       7       8       9
        #
        # we get
        #
        # ms_to_idx:
        #       0       2       2       3       3       3       5       5       8       9
        self.ms_to_idx = np.asarray(self.h5f['ms_to_idx'], dtype='int64')

        if "t_offset" in list(h5f.keys()):
            self.t_offset = int(h5f['t_offset'][()])
        else:
            self.t_offset = 0
        self.t_final = int(self.events['t'][-1]) + self.t_offset

    def get_start_time_us(self):
        return self.t_offset

    def get_final_time_us(self):
        return self.t_final

    def get_events(self, t_start_us: int, t_end_us: int) -> Dict[str, np.ndarray]:
        """Get events (p, x, y, t) within the specified time window
        Parameters
        ----------
        t_start_us: start time in microseconds
        t_end_us: end time in microseconds
        Returns
        -------
        events: dictionary of (p, x, y, t) or None if the time window cannot be retrieved
        """
        assert t_start_us < t_end_us

        # We assume that the times are top-off-day, hence subtract offset:
        t_start_us -= self.t_offset
        t_end_us -= self.t_offset

        t_start_ms, t_end_ms = self.get_conservative_window_ms(t_start_us, t_end_us)
        t_start_ms_idx = self.ms2idx(t_start_ms)
        t_end_ms_idx = self.ms2idx(t_end_ms)

        if t_start_ms_idx is None or t_end_ms_idx is None:
            # Cannot guarantee window size anymore
            return None

        events = dict()
        time_array_conservative = np.asarray(self.events['t'][t_start_ms_idx:t_end_ms_idx])
        idx_start_offset, idx_end_offset = self.get_time_indices_offsets(time_array_conservative, t_start_us, t_end_us)
        t_start_us_idx = t_start_ms_idx + idx_start_offset
        t_end_us_idx = t_start_ms_idx + idx_end_offset
        # Again add t_offset to get gps time
        events['t'] = time_array_conservative[idx_start_offset:idx_end_offset] + self.t_offset
        for dset_str in ['p', 'x', 'y']:
            events[dset_str] = np.asarray(self.events[dset_str][t_start_us_idx:t_end_us_idx])
            assert events[dset_str].size == events['t'].size
        return events

    @staticmethod
    def get_conservative_window_ms(ts_start_us: int, ts_end_us) -> Tuple[int, int]:
        """Compute a conservative time window of time with millisecond resolution.
        We have a time to index mapping for each millisecond. Hence, we need
        to compute the lower and upper millisecond to retrieve events.
        Parameters
        ----------
        ts_start_us:    start time in microseconds
        ts_end_us:      end time in microseconds
        Returns
        -------
        window_start_ms:    conservative start time in milliseconds
        window_end_ms:      conservative end time in milliseconds
        """
        assert ts_end_us > ts_start_us
        window_start_ms = math.floor(ts_start_us/1000)
        window_end_ms = math.ceil(ts_end_us/1000)
        return window_start_ms, window_end_ms

    @staticmethod
    @numba.jit(nopython=True)
    def get_time_indices_offsets(
            time_array: np.ndarray,
            time_start_us: int,
            time_end_us: int) -> Tuple[int, int]:
        """Compute index offset of start and end timestamps in microseconds
        Parameters
        ----------
        time_array:     timestamps (in us) of the events
        time_start_us:  start timestamp (in us)
        time_end_us:    end timestamp (in us)
        Returns
        -------
        idx_start:  Index within this array corresponding to time_start_us
        idx_end:    Index within this array corresponding to time_end_us
        such that (in non-edge cases)
        time_array[idx_start] >= time_start_us
        time_array[idx_end] >= time_end_us
        time_array[idx_start - 1] < time_start_us
        time_array[idx_end - 1] < time_end_us
        this means that
        time_start_us <= time_array[idx_start:idx_end] < time_end_us
        """

        assert time_array.ndim == 1

        idx_start = -1
        if time_array[-1] < time_start_us:
            # This can happen in extreme corner cases. E.g.
            # time_array[0] = 1016
            # time_array[-1] = 1984
            # time_start_us = 1990
            # time_end_us = 2000

            # Return same index twice: array[x:x] is empty.
            return time_array.size, time_array.size
        else:
            for idx_from_start in range(0, time_array.size, 1):
                if time_array[idx_from_start] >= time_start_us:
                    idx_start = idx_from_start
                    break
        assert idx_start >= 0

        idx_end = time_array.size
        for idx_from_end in range(time_array.size - 1, -1, -1):
            if time_array[idx_from_end] >= time_end_us:
                idx_end = idx_from_end
            else:
                break

        assert time_array[idx_start] >= time_start_us
        if idx_end < time_array.size:
            assert time_array[idx_end] >= time_end_us
        if idx_start > 0:
            assert time_array[idx_start - 1] < time_start_us
        if idx_end > 0:
            assert time_array[idx_end - 1] < time_end_us
        return idx_start, idx_end

    def ms2idx(self, time_ms: int) -> int:
        assert time_ms >= 0
        if time_ms >= self.ms_to_idx.size:
            return None
        return self.ms_to_idx[time_ms]

class EventReaderAbstract:
    def __init__(self, filepath: Path):
        assert filepath.is_file()
        assert filepath.name.endswith('.h5')
        self.h5f = h5py.File(str(filepath), 'r')
        self._finalizer = weakref.finalize(self, self.close_callback, self.h5f)

    @staticmethod
    def close_callback(h5f: h5py.File):
        h5f.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._finalizer()

    def __iter__(self):
        return self

    def __next__(self):
        raise NotImplementedError

class EventReader(EventReaderAbstract):
    def __init__(self, filepath: Path, dt_milliseconds: int):
        super().__init__(filepath)
        self.event_slicer = EventSlicer(self.h5f)

        self.dt_us = int(dt_milliseconds * 1000)
        self.t_start_us = self.event_slicer.get_start_time_us()
        self.t_end_us = self.event_slicer.get_final_time_us()

        self._length = (self.t_end_us - self.t_start_us)//self.dt_us

    def __len__(self):
        return self._length

    def __next__(self):
        t_end_us = self.t_start_us + self.dt_us
        if t_end_us > self.t_end_us:
            raise StopIteration
        events = self.event_slicer.get_events(self.t_start_us, t_end_us)
        if events is None:
            raise StopIteration

        self.t_start_us = t_end_us
        return events