import os
import cv2
import h5py
import torch
import imageio
import weakref
import hdf5plugin
import numpy as np
import pandas as pd

from pathlib import Path
from torch.utils.data import Dataset

from src.utils.event_proc import EventSlicer
from src.utils.flow_proc import flow_16bit_to_float

class DSECSequence(Dataset):
    def __init__(self, seq_path: Path, phase: str = 'train',  
                 timestamp_path: str = None, norm_type=None, quantile=0):
        assert seq_path.is_dir(), f"{seq_path} is not a valid directory"
        
        self.name = seq_path.name
        self.phase = phase
        self.height, self.width = 480, 640  # Constants for output dimensions
        
        # Event data paths
        self.event_slicer, self.rectify_ev_map = self._initialize_event_data(seq_path)
        self.delta_t_us = 1e5  # 100ms per frame
        
        # Load camera intrinsic
        self.intrinsic = self.load_calib(self.name)

        # Load and compute timestamps and indices
        if self.phase == 'train':
            self._load_training_data(seq_path)
        elif self.phase == 'val':
            self._load_validation_data(seq_path)
        elif self.phase == 'test':
            self._load_test_data(seq_path, timestamp_path)
        else:
            raise ValueError(f"Invalid phase: {self.phase}")
        
        self._finalizer = weakref.finalize(self, self._close_h5_file)

    def _initialize_event_data(self, seq_path: Path):
        """Load event data and rectify map from h5 files."""
        ev_dir = seq_path / 'events/left'
        self.h5f_events = h5py.File(ev_dir / 'events.h5', 'r')
        self.h5f_rectify = h5py.File(ev_dir / 'rectify_map.h5', 'r')
        rectify_map = self.h5f_rectify['rectify_map'][()]
        # rectify_map = h5py.File(ev_dir / 'rectify_map.h5', 'r')['rectify_map'][()]

        return EventSlicer(self.h5f_events), rectify_map

    def _load_training_data(self, seq_path: Path):
        """Load timestamps and paths for training phase."""
        timestamps_images = np.loadtxt(seq_path / 'images/timestamps.txt', dtype='int64')
        image_indices = np.arange(len(timestamps_images))

        starttime_flow = timestamps_images[::2][1:-1]
        endtime_flow = starttime_flow + self.delta_t_us
        self.timestamps_flow = np.stack((starttime_flow, endtime_flow), axis=1)
        self.indices = image_indices[::2][1:-1]

        keep_i = self.timestamps_flow[:, 1] < self.event_slicer.t_final
        self.timestamps_flow, self.indices = self.timestamps_flow[keep_i], self.indices[keep_i]
        self.paths_to_forward_flow = [
            seq_path / 'flow/forward' / f'{str(i).zfill(6)}.png' for i in self.indices
        ]

    def _load_validation_data(self, seq_path: Path):
        """Load timestamps and paths for validation phase."""
        self.timestamps_flow = np.loadtxt(seq_path / 'flow/forward_timestamps.txt', 
                                          delimiter=',', skiprows=1, dtype='int64')
        keep_i = self.timestamps_flow[:, 0] > self.event_slicer.t_offset
        self.timestamps_flow = self.timestamps_flow[keep_i]

        flow_files = [f for f, keep in zip(sorted(os.listdir(seq_path / 'flow/forward')), keep_i) if keep]
        self.paths_to_forward_flow = [seq_path / 'flow/forward' / f for f in flow_files]
        self.indices = [int(f.split('.')[0]) for f in flow_files]

    def _load_test_data(self, seq_path: Path, timestamp_path: str):
        """Load timestamps for test phase."""
        if timestamp_path is None:
            raise ValueError("Test timestamp path cannot be None")

        df = pd.read_csv(timestamp_path)
        self.timestamps_flow = np.stack((df['from_timestamp_us'].to_numpy(), df['to_timestamp_us'].to_numpy()), axis=1)
        self.indices = df['file_index']
        self.paths_to_forward_flow = None

    # @staticmethod
    def _close_h5_file(self):
        if hasattr(self, 'h5f_events') and self.h5f_events:
            self.h5f_events.close()
        if hasattr(self, 'h5f_rectify') and self.h5f_rectify:
            self.h5f_rectify.close()

    # def get_data_sample(self, index, flip=None):
    #     """Get a sample from the dataset."""
    #     t_start, t_end = self.timestamps_flow[index]
    #     file_index = self.indices[index]

    #     output = {
    #         'name': f'{self.name}_{str(file_index).zfill(6)}',
    #         'timestamp': torch.tensor([t_start, t_end]),
    #         'file_index': torch.tensor(file_index, dtype=torch.int64)
    #     }

    #     # Unrectified events
    #     event_data = self.event_slicer.get_events(t_start, t_end)
    #     raw_events = np.column_stack((event_data['y'], event_data['x'], event_data['t'], event_data['p']))
    #     raw_mask = (0 <= raw_events[:, 0]) & (raw_events[:, 0] < self.height) & (0 <= raw_events[:, 1]) & (raw_events[:, 1] < self.width)
    #     raw_events = raw_events[raw_mask].astype('float32')
    #     output['raw_events'] = torch.from_numpy(raw_events)
        
    #     # Rectified events
    #     x_rect, y_rect = self.rectify_events(event_data['x'], event_data['y']).T
    #     # t = (event_data['t'] - event_data['t'].min()) / (event_data['t'].max() - event_data['t'].min())
    #     # t = (event_data['t'] - event_data['t'].min()) * 1.0e-6 * 5000
    #     t = (event_data['t'] - self.timestamps_flow[0][0]) * 1.0e-6
    #     events = np.column_stack((y_rect, x_rect, t, event_data['p']))
    #     mask = (0 <= events[:, 0]) & (events[:, 0] < self.height) & (0 <= events[:, 1]) & (events[:, 1] < self.width)
    #     events = events[mask].astype('float32')
    #     output['events'] = torch.from_numpy(events)

    #     return output
    
    def get_data_sample(self, index, flip=None):
        """Get a sample from the dataset."""
        original_t_start, original_t_end = self.timestamps_flow[index]
        file_index = self.indices[index]

        output = {
            'name': f'{self.name}_{str(file_index).zfill(6)}',
            'timestamp': torch.tensor([original_t_start, original_t_end]),
            'file_index': torch.tensor(file_index, dtype=torch.int64)
        }

        # 目标事件数量
        target_num = 1500000  # 1.5M events

        # 转换为相对于event_slicer的时间
        t_start_rel = original_t_start - self.event_slicer.t_offset
        t_end_rel = original_t_end - self.event_slicer.t_offset

        # 计算初始毫秒窗口
        t_start_ms = t_start_rel // 1000
        t_end_ms = t_end_rel // 1000

        # 获取初始索引范围
        start_idx = self.event_slicer.ms2idx(t_start_ms)
        end_idx = self.event_slicer.ms2idx(t_end_ms)

        # 处理索引越界情况
        if start_idx is None:
            start_idx = 0
        if end_idx is None:
            end_idx = len(self.event_slicer.events['t'])

        current_num = end_idx - start_idx

        # 动态调整索引范围
        if current_num < target_num:
            # 扩展索引范围
            insufficient = target_num - current_num
            start_idx = max(start_idx - insufficient // 2, 0)
            end_idx = min(end_idx + insufficient - insufficient//2, len(self.event_slicer.events['t']))
            current_num = end_idx - start_idx
            
            # 确保最终数量不小于目标值
            if current_num < target_num:
                end_idx = min(start_idx + target_num, len(self.event_slicer.events['t']))
        elif current_num > target_num:
            # 截断到目标数量
            start_idx = max(end_idx - target_num, 0)

        # 计算新时间窗口
        try:
            adjusted_t_start = self.event_slicer.events['t'][start_idx] + self.event_slicer.t_offset
            adjusted_t_end = self.event_slicer.events['t'][end_idx-1] + self.event_slicer.t_offset
        except IndexError:
            adjusted_t_start = original_t_start
            adjusted_t_end = original_t_end

        # 获取调整后的事件数据
        event_data = self.event_slicer.get_events(adjusted_t_start, adjusted_t_end)
        if event_data is None:
            event_data = {'x': np.array([]), 'y': np.array([]), 't': np.array([]), 'p': np.array([])}

        # 最终数量调整
        n_events = len(event_data['x'])
        if n_events > target_num:
            # 保留最后1.5M事件
            event_data = {k: v[-target_num:] for k, v in event_data.items()}

        # 处理原始事件
        raw_events = np.column_stack((event_data['y'], event_data['x'], event_data['t'], event_data['p']))
        raw_mask = (0 <= raw_events[:, 0]) & (raw_events[:, 0] < self.height) & \
                (0 <= raw_events[:, 1]) & (raw_events[:, 1] < self.width)
        raw_events = raw_events[raw_mask].astype('float32')
        output['raw_events'] = torch.from_numpy(raw_events)

        # 处理矫正事件
        x_rect, y_rect = self.rectify_events(event_data['x'], event_data['y']).T
        t = (event_data['t'] - event_data['t'].min()) * 1.0e-6
        events = np.column_stack((y_rect, x_rect, t, event_data['p']))
        mask = (0 <= events[:, 0]) & (events[:, 0] < self.height) & \
            (0 <= events[:, 1]) & (events[:, 1] < self.width)
        events = events[mask].astype('float32')
        output['events'] = torch.from_numpy(events)
        return output

    def load_calib(self, sequence_name) -> dict:
        """Load calibration file.

        Outputs:
            (dict) ... {"K": camera_matrix, "D": distortion_coeff}
                camera_matrix (np.ndarray) ... [3 x 3] matrix.
                distortion_coeff (np.array) ... [4] array.
        """
        intrinsics_mat = []
        
        if(sequence_name == "interlaken_00_b"):
            intrinsics_mat = np.array(
                [
                    [569.7632987676102, 0, 335.0999870300293],
                    [0, 569.7632987676102, 221.23667526245117],
                    [0, 0, 1],
                ],
                dtype=np.float32,
            )
        elif(sequence_name == "interlaken_01_a"):
            intrinsics_mat = np.array(
                [
                    [569.7632987676102, 0, 335.0999870300293],
                    [0, 569.7632987676102, 221.23667526245117],
                    [0, 0, 1],
                ],
                dtype=np.float32,
            )
        elif(sequence_name == "thun_01_a"):
            intrinsics_mat = np.array(
                [
                    [569.7632987676102, 0, 335.0999870300293],
                    [0, 569.7632987676102, 221.23667526245117],
                    [0, 0, 1],
                ],
                dtype=np.float32,
            )
        elif(sequence_name == "thun_01_b"):
            intrinsics_mat = np.array(
                [
                    [569.7632987676102, 0, 335.0999870300293],
                    [0, 569.7632987676102, 221.23667526245117],
                    [0, 0, 1],
                ],
                dtype=np.float32,
            )
        elif(sequence_name == "zurich_city_12_a"):
            intrinsics_mat = np.array(
                [
                    [583.3081203392971, 0, 336.83414459228516],
                    [0, 583.3081203392971, 220.91131019592285],
                    [0, 0, 1],
                ],
                dtype=np.float32,
            )
        elif(sequence_name == "zurich_city_14_c"):
            intrinsics_mat = np.array(
                [
                    [583.3081203392971, 0, 336.83414459228516],
                    [0, 583.3081203392971, 220.91131019592285],
                    [0, 0, 1],
                ],
                dtype=np.float32,
            )
        elif(sequence_name == "zurich_city_15_a"):
            intrinsics_mat = np.array(
                [
                    [576.0330202256714, 0, 335.0866508483887],
                    [0, 576.0330202256714, 221.45818328857422],
                    [0, 0, 1],
                ],
                dtype=np.float32,
            )
        else:
            raise ValueError
        return intrinsics_mat
    
    def __len__(self):
        return len(self.timestamps_flow)

    def rectify_events(self, x, y):
        """Rectify event coordinates."""
        return self.rectify_ev_map[y, x]

    def __getitem__(self, idx):
        return self.get_data_sample(idx)

    @staticmethod
    def get_disparity_map(filepath: Path):
        """Load disparity map."""
        assert filepath.is_file(), f"{filepath} is not a valid file"
        return cv2.imread(str(filepath), cv2.IMREAD_ANYDEPTH).astype('float32') / 256

    @staticmethod
    def load_flow(flowfile: Path):
        """Load optical flow from PNG file."""
        assert flowfile.exists(), f"Flow file {flowfile} not found"
        assert flowfile.suffix == '.png', "Flow file must be a PNG"
        flow_16bit = imageio.imread(str(flowfile), format='PNG-FI')
        return flow_16bit_to_float(flow_16bit)

def pad_events(events, max_length):
    padded = torch.zeros((max_length, 5), dtype=events.dtype)
    padded[:len(events), :4] = events
    padded[:len(events), 4] = 1
    return padded

def sequence_collate_fn(batch):
    batched_data = {
        'events': [], 
        'raw_events': [],
        'timestamp': [], 
        'file_index': []
    }

    event_types = ('events',) if 'events' in batch[0] else ('pos_events', 'neg_events')
    max_events = {et: 0 for et in event_types}
    max_raw_events = 0  
    all_have_gt_flow = all('forward_flow' in sample for sample in batch)

    for sample in batch:
        for et in event_types:
            max_events[et] = max(max_events[et], len(sample[et]))
        max_raw_events = max(max_raw_events, len(sample['raw_events']))

    if len(event_types) > 1:
        batched_data['num_pos_events'] = max_events['pos_events']

    if all_have_gt_flow:
        batched_data['forward_flow'] = []
        batched_data['flow_valid'] = []

    for sample in batch:
        if len(event_types) == 1:
            padded_events = pad_events(sample['events'], max_events['events'])
        else:
            pos_padded = pad_events(sample['pos_events'], max_events['pos_events'])
            neg_padded = pad_events(sample['neg_events'], max_events['neg_events'])
            padded_events = torch.cat((pos_padded, neg_padded), dim=0)
        raw_padded = pad_events(sample['raw_events'], max_raw_events)
        
        batched_data['events'].append(padded_events)
        batched_data['raw_events'].append(raw_padded)
        batched_data['timestamp'].append(sample['timestamp'])
        batched_data['file_index'].append(sample['file_index'])

        if all_have_gt_flow:
            batched_data['forward_flow'].append(sample['forward_flow'])
            batched_data['flow_valid'].append(sample['flow_valid'])

    batched_data['events'] = torch.stack(batched_data['events'], dim=0)
    batched_data['raw_events'] = torch.stack(batched_data['raw_events'], dim=0)
    batched_data['timestamp'] = torch.stack(batched_data['timestamp'], dim=0)
    batched_data['file_index'] = torch.stack(batched_data['file_index'], dim=0)

    if all_have_gt_flow:
        batched_data['forward_flow'] = torch.stack(batched_data['forward_flow'], dim=0)
        batched_data['flow_valid'] = torch.stack(batched_data['flow_valid'], dim=0)

    return batched_data

