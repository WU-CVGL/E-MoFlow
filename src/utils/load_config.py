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
    parser = argparse.ArgumentParser(description="configs for E-MoFlow")

    parser.add_argument('--config', type=str, default='./configs/dsec/interlaken_00_b.yaml', 
                        help="Path to the config file.")
    parser.add_argument('--gpu', type=int, default=0,
                        help='index of gpu to use')
    args = parser.parse_args()
    return args

