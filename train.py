import torch 
import torchdiffeq
import numpy as np

from src.event_data import EventStreamData

if __name__ == "__main__":
    eventstream = EventStreamData(
        data_path = "/home/liwenpu-cvgl/events/002000.txt", t_start = 0, t_end = 1, 
        H=480, W=640, color_event=False, event_thresh=1, device='cpu'
    )

    eventstream.stack_event_images(1)
    eventstream.visuailize_event_images()
   
    



