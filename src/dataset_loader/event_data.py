import os
import math
import h5py
import torch
import numba
import weakref
import cv2 as cv
import numpy as np

from pathlib import Path
from typing import Union, Dict, Tuple

class EventData():
    def __init__(self, data: Union[np.ndarray, torch.Tensor], H, W, device: torch.device):
        if isinstance(data, np.ndarray):
            assert (
                data.shape[1] == 4 and data.ndim == 2
            ), f"[ERROR] {self} Can not be recognized as an event stream due to incorrect format!"
        elif isinstance(data, torch.Tensor):
            assert (
               data.shape[1] == 4 and data.dim() == 2 
            ), f"[ERROR] {self} Can not be recognized as an event stream due to incorrect format!"

        self.events = data
        self.device = device
        self.H = H
        self.W = W
        self.event_image = None
    
    def __len__(self):
        return self.events[0]

    def to_event_image(self) -> Union[np.ndarray, torch.tensor]:
        ts = self.events[:, 0]
        xs = self.events[:, 1]
        ys = self.events[:, 2]
        ps = self.events[:, 3]
        ps[ps == 0] = -1
        out = np.zeros((self.H, self.W))

        if self.device.type == "cpu":
            for i in range(xs.shape[0]):
                x, y, p = xs[i], ys[i], ps[i]
                out[int(y), int(x)] += int(p)
            self.event_image = out
        elif self.device.type == "cuda":
            with torch.no_grad():
                # spare tensor
                indices_array = np.array([ys, xs])
                indices_tensor = torch.tensor(indices_array, dtype = torch.long)
                values = torch.tensor(ps, dtype = torch.float32)
                size = torch.Size([out.shape[0], out.shape[1]])
                out_sparse = torch.sparse_coo_tensor(indices_tensor, values, size)
                # dense tensor
                out_tensor = torch.from_numpy(out).to(self.device)
                out_tensor += out_sparse.to_dense().to(self.device)
                self.event_image = out_tensor.cpu().numpy()
        return self.event_image

class EventStreamData():
    def __init__(self, data_path, t_start, t_end, H, W, color_event, event_thresh, device: torch.device):
        self.data_path = data_path
        self.t_start = t_start
        self.t_end = t_end
        self.H, self.W = H, W
        self.color_event = color_event
        self.event_thresh = event_thresh
        self.device = device

        self.events = self.load_events()
        
    def load_events(self):
        _, file_extension = os.path.splitext(self.data_path)
        if file_extension == ".txt":
            events = np.loadtxt(self.data_path)
        elif file_extension == ".npy":
            events = np.load(self.data_path, allow_pickle=True)

        events[: ,0] = events[:, 0] - events[0, 0]
        events = events[(events[:, 0] > self.t_start) & (events[:, 0] < self.t_end)]
        events[: ,0] = (events[: ,0] - self.t_start) / (self.t_end - self.t_start) * 2 - 1 # Normalize event timestampes to [-1, 1]

        if self.H > self.W:
            self.H, self.W = self.W, self.H
            events = events[:, [0, 2, 1, 3]]
            
        if events.shape[0] == 0:
            raise ValueError(f'No events in [{self.t_start}, {self.t_end}]!')
        print(f'Loaded {events.shape[0]} events in [{self.t_start}, {self.t_end}] ...')
        print(f'First event: {events[0]}')
        print(f'Last event: {events[-1]}')
        return events
    
    def stack_event_images(self, num_frames):
        print(f'Stacking {num_frames} event frames from {self.events.shape[0]} events ...')
        event_chunks = np.array_split(self.events, num_frames)
        event_images, event_timestamps = [], []
        for i, event_chunk in enumerate(event_chunks):
            event_image = EventData(event_chunk, self.H, self.W, device = self.device).to_event_image()
            # if self.color_event:
            #     event_image = quad_bayer_to_rgb_d2(event_image)
            event_image *= self.event_thresh
            event_images.append(event_image)
            event_timestamps.append((event_chunk[0, 0] + event_chunk[-1, 0]) / 2)
        
        event_images = np.stack(event_images, axis=0).reshape(num_frames, self.H, self.W)
        self.event_images = torch.as_tensor(event_images).float().to(self.device)
        event_image_timestamps = np.stack(event_timestamps, axis=0).reshape(num_frames, 1)
        self.event_image_timestamps = torch.as_tensor(event_image_timestamps).float().to(self.device)
    
    def visuailize_event_images(self):
        rgb_image = np.full((self.H, self.W, 3), fill_value=255,dtype='uint8')
        for i, event_image in enumerate(self.event_images):
            event_image = event_image.cpu().detach().numpy()
            print(event_image.shape)
            rgb_image[event_image==0]=[255,255,255]
            rgb_image[event_image<=(-1*self.event_thresh)]=[255,0,0]
            rgb_image[event_image>=(1*self.event_thresh)]=[0,0,255]
            cv.imwrite(f"event_image_{i}.png", rgb_image)

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