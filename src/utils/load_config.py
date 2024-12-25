import yaml
import argparse

def load_yaml_config(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    logger_config = data['logger']
    variables = {
        'project': logger_config['project'],
        'expname': logger_config['expname']
    }
    
    for key, value in logger_config.items():
        if isinstance(value, str):
            temp_value = value  
            for var_name, var_value in variables.items():
                temp_value = temp_value.replace(f'${{project}}', variables['project'])
                temp_value = temp_value.replace(f'${{expname}}', variables['expname'])
            logger_config[key] = temp_value
    
    return data

def parse_args():
    parser = argparse.ArgumentParser(description="configs for NeuEF")

    parser.add_argument('--config', type=str, default='configs/synthetic.yaml', 
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

