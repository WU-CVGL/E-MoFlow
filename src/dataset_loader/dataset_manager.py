import logging

from .simple_test.reader import SimpleTest

logger = logging.getLogger(__name__)

def get_dataset(dataset_name, data_config):
    if dataset_name == "simple_test":
        logger.info(f"Use simple_test datset.")
        return SimpleTest(**data_config)
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