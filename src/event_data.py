import os
import torch
# import numba
import cv2 as cv
import numpy as np

from typing import Union

class EventData():
    def __init__(self, data: Union[np.ndarray, torch.Tensor], H, W, device = "cuda"):
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

        if self.device == "cpu":
            for i in range(xs.shape[0]):
                x, y, p = xs[i], ys[i], ps[i]
                out[int(y), int(x)] += int(p)
            self.event_image = out
        elif self.device == "cuda":
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
    def __init__(self, data_path, t_start, t_end, H=260, W=346, color_event=True, event_thresh=1, device='cuda'):
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
        timestamps = np.stack(event_timestamps, axis=0).reshape(num_frames, 1)
        self.timestamps = torch.as_tensor(timestamps).float().to(self.device)
    
    def visuailize_event_images(self):
        rgb_image = np.full((self.H, self.W, 3), fill_value=255,dtype='uint8')
        for i, event_image in enumerate(self.event_images):
            event_image = event_image.cpu().detach().numpy()
            print(event_image.shape)
            rgb_image[event_image==0]=[255,255,255]
            rgb_image[event_image<=(-1*self.event_thresh)]=[255,0,0]
            rgb_image[event_image>=(1*self.event_thresh)]=[0,0,255]
            cv.imwrite(f"event_image_{i}.png", rgb_image)