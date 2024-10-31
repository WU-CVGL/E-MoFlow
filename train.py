import os
import sys
import time
import random
import logging
import tensorhue
import cv2 as cv
import numpy as np

import torch 
import torch.optim as optim

from tqdm import tqdm
from scipy.ndimage import zoom

from src import load_config
from src.loss import focus
from src.wandb import WandbLogger
from src.model.inr import EventFlowINR
from src.model.warp import NeuralODEWarp
from src.event_data import EventStreamData
from src.dataset_loader import dataset_manager
from src.event_image_converter import EventImageConverter

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
    warp_config = config["warp"]

    # wandb
    wandb_logger = WandbLogger(config)

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

    # prepare data to test
    start_indice, end_indice = config["test"]["file_idx"]
    test_event_file_paths = dataset.txt_files_path[start_indice:end_indice+1]
    test_event_data_list = [np.loadtxt(file) for file in test_event_file_paths]
    test_events = torch.from_numpy(np.vstack(test_event_data_list))

    mask = (0 <= test_events[:, 1]) & (test_events[:, 1] < dataset.weight) & \
            (0 <= test_events[:, 2]) & (test_events[:, 2] < dataset.hight)
    test_events = test_events[mask]
    test_events_sorted, sort_indices = torch.sort(test_events[:, 0])
    test_events_sorted = test_events[sort_indices]

    test_events_norm = test_events_sorted.clone()
    test_events_norm[:, 0] = (test_events_sorted[:, 0] - dataset.event_start_time) / (dataset.event_end_time - dataset.event_start_time) 
    test_events_norm[:, 1] = test_events_sorted[:, 1] / dataset.weight
    test_events_norm[:, 2] = test_events_sorted[:, 2] / dataset.hight
    test_batch_txy = test_events_norm[:, :-1].float().to(device)

    # event2img converter
    image_size = (data_config["weight"], data_config["hight"])
    converter = EventImageConverter(image_size)

    # create model
    flow_field = EventFlowINR(model_config).to(device)
    warpper = NeuralODEWarp(flow_field, device, **warp_config)

    # create optimizer
    optimizer = optim.Adam(flow_field.parameters(), lr=1e-6)

    # display origin test data
    with torch.no_grad():
        iwe = converter.create_iwe(test_events_sorted[:, [1,2,0,3]])
        wandb_logger.write_img("iwe", iwe.T.detach().cpu().numpy() * 255)

    # train
    num_epochs = 200
    for i in range(num_epochs):
        for idx, sample in enumerate(tqdm(loader, desc=f"Tranning {i} epoch", leave=True)):
            # get batch data
            events = sample["events"].squeeze(0)
            events_norm = sample["events_norm"].squeeze(0)
            timestamps = sample["timestamps"].squeeze(0)
            batch_txy = events_norm[:, :-1].float().to(device)

            # iwe = converter.create_iwe(events[:, [1,2,0,3]])
            # cv.imwrite(f"iwe_{idx}.png", (iwe.T.detach().cpu().numpy()) * 255)

            # get t_ref
            ref = warpper.get_reference_time(batch_txy, warp_config["tref_setting"])

            # odewarp 
            warped_batch_txy = warpper.warp_events(batch_txy, ref)

            # create image warped event           
            polarity = events[:, 3].unsqueeze(1).to(device)
            warped_events_xytp = torch.cat((warped_batch_txy[:, [1,2,0]], polarity), dim=1)
            warped_events_xytp[:, :2] *= torch.Tensor(image_size).to(device)
            iwe = converter.create_iwe(warped_events_xytp)
            
            # loss
            var_loss = focus.calculate_focus_loss(iwe, loss_type='variance')
            grad_loss = focus.calculate_focus_loss(iwe, loss_type='gradient_magnitude', norm='l1')
            total_loss = var_loss + grad_loss
            wandb_logger.write("var_loss", var_loss.item())
            wandb_logger.write("grad_loss", grad_loss.item())
            wandb_logger.write("train_loss", total_loss.item())

            # optimize
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # log
            wandb_logger.update_buffer()
        # print(total_loss.item())
        tqdm.write(
            f"[LOG] Epoch {i} Total Loss: {total_loss.item()}, Var Loss: {var_loss.item()}, Grad Loss: {grad_loss.item()},"
        )

        # test
        with torch.no_grad():
            test_ref = warpper.get_reference_time(test_batch_txy, "max")
            test_warped_batch_txy = warpper.warp_events(test_batch_txy, test_ref)
            # create image warped event           
            test_polarity = test_events_sorted[:, 3].unsqueeze(1).to(device)
            test_warped_events_xytp = torch.cat((test_warped_batch_txy[:, [1,2,0]], test_polarity), dim=1)
            test_warped_events_xytp[:, :2] *= torch.Tensor(image_size).to(device)
            iwe = converter.create_iwe(test_warped_events_xytp)
            wandb_logger.write_img("iwe", iwe.T.detach().cpu().numpy() * 255)

            # iwe_numpy = iwe.T.detach().cpu().numpy()
            # tensorhue.viz(zoom(iwe_numpy, (1/8,1/8)))
            # cv.imwrite(f"iwe_{j}.png", (iwe.T.detach().cpu().numpy()) * 255)
            # print(warped_events[0][0].item(), warped_events[-1][0].item(), ref["t_ref"].item())
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
    #     data_path = "/home/liwenpu-cvgl/events/000550.txt", t_start = 0, t_end = 1, 
    #     H=480, W=640, color_event=False, event_thresh=1, device=device
    # )
    # eventstream.stack_event_images(1)
    # eventstream.visuailize_event_images()

    # eventstream.events = eventstream.events[:, [1,2,0,3]]
    # iwe = converter.create_iwe(eventstream.events)
    # # print(iwe.shape)
    # cv.imwrite(f"iwe.png", iwe.T)
    # tensorhue.viz(zoom(iwe.T, (1/8,1/8)))
