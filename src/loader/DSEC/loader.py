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
    
    def adaptive_spatiotemporal_sampling(self, events, max_events, device='cuda'):
        if len(events) <= max_events:
            return events
        
        events_gpu = torch.from_numpy(events).to(device)
        n_events = len(events_gpu)
        
        spatial_range = (events_gpu[:, :2].max(dim=0)[0] - events_gpu[:, :2].min(dim=0)[0])
        temporal_range = events_gpu[:, 2].max() - events_gpu[:, 2].min()
    
        spatial_bins = max(10, min(100, int(torch.sqrt(spatial_range.prod()).item() / 10)))
        time_resolution = 0.01  # 10ms
        temporal_bins = max(5, min(50, int(temporal_range / time_resolution)))
        
        return self.spatiotemporal_density_sampling(
            events, max_events, spatial_bins, temporal_bins, device
        )
    
    def spatiotemporal_density_sampling(self, events, max_events, spatial_bins=50, temporal_bins=20, device='cuda'):
        """ Perform spatiotemporal density sampling on events."""
        if len(events) <= max_events:
            return events
        
        events_gpu = torch.from_numpy(events).to(device)
        n_events = len(events_gpu)
        
        # spatial bin
        x_max, y_max = events_gpu[:, 1].max(), events_gpu[:, 0].max()
        x_edges = torch.linspace(0, x_max, spatial_bins + 1, device=device)
        y_edges = torch.linspace(0, y_max, spatial_bins + 1, device=device)
        
        # temporal bin
        t_min, t_max = events_gpu[:, 2].min(), events_gpu[:, 2].max()
        t_edges = torch.linspace(t_min, t_max, temporal_bins + 1, device=device)
        
        # calc spatial-temporal bin indices
        x_bin_idx = torch.searchsorted(x_edges[1:], events_gpu[:, 1])
        y_bin_idx = torch.searchsorted(y_edges[1:], events_gpu[:, 0])
        t_bin_idx = torch.searchsorted(t_edges[1:], events_gpu[:, 2])
        
        # calc density in each bin
        bin_ids = (t_bin_idx * spatial_bins * spatial_bins + 
                y_bin_idx * spatial_bins + 
                x_bin_idx)
        unique_bins, inverse_indices, counts = torch.unique(
            bin_ids, return_inverse=True, return_counts=True
        )
        
        # calc density weights
        total_bins = spatial_bins * spatial_bins * temporal_bins
        target_density = n_events / total_bins
        densities = counts[inverse_indices].float()
        density_weights = 1.0 / (1.0 + torch.abs(densities - target_density) / (target_density + 1e-6))
        
        # sample
        if density_weights.sum() > 0:
            weights = density_weights / density_weights.sum()
            selected_indices = torch.multinomial(weights, max_events, replacement=False)
        else:
            selected_indices = torch.randperm(n_events, device=device)[:max_events]
        
        return events[selected_indices.cpu().numpy()]

    
    def get_data_sample(self, index, flip=None):
        """Get a sample from the dataset."""
        t_start, t_end = self.timestamps_flow[index]
        t_step = (t_end - t_start) 
        t_end = t_start + t_step
        file_index = self.indices[index]

        output = {
            'name': f'{self.name}_{str(file_index).zfill(6)}',
            'timestamp': torch.tensor([t_start, t_end]),
            'file_index': torch.tensor(file_index, dtype=torch.int64)
        }

        # Unrectified events
        event_data = self.event_slicer.get_events(t_start, t_end)
        raw_events = np.column_stack((event_data['y'], event_data['x'], event_data['t'], event_data['p']))
        raw_mask = (0 <= raw_events[:, 0]) & (raw_events[:, 0] < self.height) & (0 <= raw_events[:, 1]) & (raw_events[:, 1] < self.width)
        raw_events = raw_events[raw_mask].astype('float32')
        # output['raw_events'] = torch.from_numpy(raw_events[raw_events[:, 3] == 1])
        output['raw_events'] = torch.from_numpy(raw_events)
        
        # Rectified events
        x_rect, y_rect = self.rectify_events(event_data['x'], event_data['y']).T
        # t = (event_data['t'] - event_data['t'].min()) / (event_data['t'].max() - event_data['t'].min())
        # t = (event_data['t'] - event_data['t'].min()) * 1.0e-6 * 5000
        t = (event_data['t'] - self.timestamps_flow[0][0]) * 1.0e-6
        events = np.column_stack((y_rect, x_rect, t, event_data['p']))
        mask = (0 <= events[:, 0]) & (events[:, 0] < self.height) & (0 <= events[:, 1]) & (events[:, 1] < self.width)
        events = events[mask].astype('float32')
        # output['events'] = torch.from_numpy(events[events[:, 3] == 1])
        output['events'] = torch.from_numpy(events)
        n_events = events.shape[0]   
        max_limit_events = 1500000
        if n_events >= max_limit_events:
            events = self.adaptive_spatiotemporal_sampling(events, max_limit_events)
            
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

