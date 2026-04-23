import yaml
import os
methods = ['GF', 'AA', 'AA-l2']

def train(path_yaml: str):

    with open(path_yaml,'r') as file:
        try:
            config_data = yaml.safe_load(file)
        except yaml.YAMLError as e:
            print(f' Error reading YAML file: {e}')
    # Avoid preallocation and fix a device
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = config_data['config']['preallocate']
    os.environ['CUDA_VISIBLE_DEVICES'] = config_data['config']['device']

    




    

