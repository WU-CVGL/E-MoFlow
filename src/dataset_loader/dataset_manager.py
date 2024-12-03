import torch
import logging
import numpy as np

from torch.utils.data import DataLoader

from .base import DatasetBase
from .simple_test.reader import SimpleTestDataset
from .simple_test.provider import SimpleTestDataProvider
from .synthetic_figure.reader import SyntheticFigureDataset
from .synthetic_figure.provider import SyntheticFigureDataProvider

logger = logging.getLogger(__name__)

def get_dataset(dataset_name, data_config):
    if dataset_name == "SimpleTest":
        logger.info(f"Use SimpleTest datset.")
        return SimpleTestDataset(**data_config)
    elif dataset_name == "SyntheticFigure":
        logger.info(f"Use SyntheticFigure datset.")
        return SyntheticFigureDataset(**data_config)
    elif dataset_name == "DSEC":
        pass
    elif dataset_name == "MVSEC":
        pass
    elif dataset_name == "EVIMO2":
        pass
    elif dataset_name == "TUM-VIE":
        pass
    else:
        logger.error(f"Invaild dataset: {dataset_name}")
        raise ValueError(f"Invaild dataset: {dataset_name}")
    
def create_loader(dataset: DatasetBase, config: dict):
    if dataset.name == "SimpleTest":
        dataprovider = SimpleTestDataProvider(dataset)
        logger.info(f"The loader for {dataset.name} is created.")
    elif dataset.name == "SyntheticFigure":
        dataprovider = SyntheticFigureDataProvider(dataset)
        logger.info(f"The loader for {dataset.name} is created.")
    else:
        e = f"Can't creat loader for {dataset.name}."
        logger.error(e)
        raise ValueError(e)        
    
    return DataLoader(
        dataprovider, 
        batch_size=config["batch_size"], 
        num_workers=config["num_workers"], 
        shuffle=config["shuffle"]
    ), dataprovider