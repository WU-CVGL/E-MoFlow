import torch 
import tensorhue
import torchdiffeq
import numpy as np

from src import load_config
from src.model.inr import EventFlowINR
from src.dataloader.event_data import EventStreamData

if __name__ == "__main__":
    # load configs
    config = load_config.load_yaml_config('configs/test.yaml')
    args = load_config.parse_args(config)
    
    # detect cuda
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    # create eventstream
    eventstream = EventStreamData(
        data_path = args.data_path, t_start = 0, t_end = 1, 
        H=480, W=640, color_event=False, event_thresh=1, device=device
    )

    test_coord = torch.tensor([[1,2,3],[4,5,6],[7,8,9],[10,11,12]]).to(device)
    tensorhue.viz(test_coord.cpu().detach().numpy())
    eventflow = EventFlowINR().to(device)
    out = eventflow.forward(test_coord, args)
    tensorhue.viz(out.cpu().detach().numpy())


   
    



