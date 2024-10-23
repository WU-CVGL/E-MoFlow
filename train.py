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
from src.model.warp import NeuralODEWarp
from src.dataset_loader import dataset_manager
# from src.dataset_loader.event_data import EventStreamData

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

    # create model
    flow_field = EventFlowINR(model_config).to(device)
    warpper = NeuralODEWarp(flow_field, device=device)

    # train
    num_epochs = 1
    for i in range(num_epochs):
        for sample in tqdm(loader, desc=f"Tranning {i} epoch", leave=True):
            # get batch data
            events = sample["events"].squeeze(0)
            events_norm = sample["events_norm"].squeeze(0)
            timestamps = sample["timestamps"].squeeze(0)
            batch_txy = events_norm[:, :-1].float().to(device)

            # get t_ref
            ref = warpper.get_reference_time(batch_txy)
            warpper.warp_events(batch_txy, **ref)
            # batch_xy0, batch_t0 = batch_txy[:, 1:], batch_txy[:, 0] 
            # t_ref = torch.max(batch_t0) + (1 - torch.max(batch_t0)) * torch.rand(1).to(device)
            # for xy0, t0 in zip(batch_xy0, batch_t0):
            #     distance = torch.abs(t_ref - t0)
            #     num_eval = max(2, int(distance / 1))  
            #     eval_t = torch.linspace(t0.item(), t_ref.item(), num_eval).to(device)
            #     # print(t0.shape)
            #     pred_y = torchdiffeq.odeint_adjoint(flow_field, xy0, eval_t).to(device)
            #     print(pred_y.device)
            # print("1 batch down")
                # print(xy0.shape)
            # print(f"max: {torch.max(t_0).item()}")
            # print(f"t_ref: {batch_t0[-1].item()}")
            # print(value_0[0][0])
            # pred_y = torchdiffeq.odeint_adjoint(flow_field, )
            # out = flow_field.forward(xyt_coordinate, model_config)
            # print(out.shape)
            # logger.info(f"start: {txy[0][0].tolist()}, end: {txy[-1][0].tolist()}")
            # time.sleep(0.5)
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
    # out = neu_flow_field.forward(test_coord, model_config)
    # print(out)
    # tensorhue.viz(out.cpu().detach().numpy())
