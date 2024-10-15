import yaml
import argparse

def load_yaml_config(file_path):
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def parse_args():
    parser = argparse.ArgumentParser(description="configs for NeuEF")

    parser.add_argument('--config', type=str, default='configs/test.yaml', 
                        help="Path to the config file.")
    parser.add_argument('--gpu', type=int, default=0,
                        help='index of gpu to use')

    args = parser.parse_args()
    return args
    # parser.add_argument('--data_path', type=str, default=config.get('data_path', None), 
    #                     help='the path to event data')
    # parser.add_argument('--dataset', type=str, default=config.get('data_path', None), 
    #                     help='the path to event data')
    # parser.add_argument('--gpu', type=int, default=config.get('gpu', 0),
    #                     help='index of gpu')
    # parser.add_argument('--pe', type=int, default=config.get('pe', 0),
    #                     help='set 0 for default positional encoding, -1 for none')
    # parser.add_argument('--multires', type=int, default=config.get('multires', 10),
    #                     help='log2 of max freq for positional encoding (3D location)')

