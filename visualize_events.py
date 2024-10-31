import torch
import cv2 as cv
import numpy as np

from src.event_data import EventData

folder_path = "/home/liwenpu-cvgl/events"
file_names = [f'{folder_path}/{i:06d}.txt' for i in range(20)]
# file_path = os.path.join("file_names", file_names)
event_data_list = [np.loadtxt(file) for file in file_names]
all_events = torch.tensor(np.vstack(event_data_list))

events = EventData(all_events, 480, 640, torch.device("cuda"))
event_image = events.to_event_image()
print(event_image.shape)
rgb_image = np.full((480, 640, 3), fill_value=255,dtype='uint8')
rgb_image[event_image==0]=[255,255,255]
rgb_image[event_image<= -1]=[255,0,0]
rgb_image[event_image>= 1]=[0,0,255]
cv.imwrite(f"event_image_10.png", rgb_image)

