import os
import sys
import time
import torch 
import random
import logging
import tensorhue
import torchdiffeq
import numpy as np

from tqdm import tqdm
from src import load_config
from src.model.inr import EventFlowINR
from src.dataset_loader import dataset_manager
from src.dataset_loader.event_data import EventStreamData

# log
logging.basicConfig(
    handlers=[
        logging.FileHandler(f"main.log", mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

def fix_random_seed(seed_idx=666) -> None:
    random.seed(seed_idx)
    np.random.seed(seed_idx)
    torch.manual_seed(seed_idx)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_idx)
        torch.cuda.manual_seed_all(seed_idx)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    # load configs
    args = load_config.parse_args()
    config = load_config.load_yaml_config(args.config)
    data_config = config["data"]
    model_config = config["model"]

    # detect cuda
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    logger.info(f"Use device: {device}")

    # fix seed
    fix_random_seed()

    # load dataset 
    dataset_name = data_config.pop('dataset_name')
    dataloader_config = data_config.pop('loader')
    dataset = dataset_manager.get_dataset(dataset_name, data_config)
    loader = dataset_manager.create_loader(dataset, dataloader_config)

    num_epochs = 1
    for i in range(num_epochs):
        for sample in tqdm(loader, desc=f"Tranning {i} epoch", leave=True):
            sample["events"] = sample["events"].float().to(device)
            print(sample["events"].dtype)
            time.sleep(1)
            # logger.info(f"data(x y t p): {sample["events"]}")
    # print(dataset.data_num)
    # create eventstream
    # eventstream = EventStreamData(
    #     data_path = config["data_path"], t_start = 0, t_end = 1, 
    #     H=480, W=640, color_event=False, event_thresh=1, device=device
    # )
    # eventstream.stack_event_images(1)
    # eventstream.visuailize_event_images()

    # test_coord = torch.tensor([[1,2,3],[4,5,6],[7,8,9],[10,11,12]]).to(device)
    # tensorhue.viz(test_coord.cpu().detach().numpy())
    # neu_flow_field = EventFlowINR().to(device)
    # out = neu_flow_field.forward(test_coord, args)
    # print(out)
    # tensorhue.viz(out.cpu().detach().numpy())
