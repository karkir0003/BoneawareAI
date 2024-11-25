import os
import configparser

# IMPORT MODELS HERE
from models.densenet import DenseNet
# from models.resnet import ResNet
# from models.vgg import VGG

def model_config_parser():
    config = configparser.ConfigParser()
    # Point to the main directory
    main_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    config_path = os.path.join(main_dir, 'models.ini')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    config.read(config_path)
    return config, config_path

def get_hyperparameters():
    config, config_path = model_config_parser()
    if 'hyperparams' not in config:
        raise KeyError(f"'hyperparameters' section not found in config file at {config_path}.")
    hyperarams = config['hyperparams']
    lr = float(hyperarams['lr'])
    weight_decay = float(hyperarams['weight_decay'])
    num_epochs = int(hyperarams['num_epochs'])
    step_size = int(hyperarams['lr_scheduler_step_size'])
    gamma = float(hyperarams['lr_scheduler_gamma'])
    return lr, weight_decay, num_epochs, step_size, gamma

def get_model(model_name, device):
    config, config_path = model_config_parser()
    model = None
    if model_name == 'densenet':
        if 'densenet' not in config:
            raise KeyError(f"'densenet' section not found in config file at {config_path}.")
        model = return_densenet(config['densenet'], device)
    elif model_name == 'resnet':
        if 'resnet' not in config:
            raise KeyError(f"'resnet' section not found in config file at {config_path}.")
        # model = return_resnet(config['resnet'], device)
    elif model_name == 'vgg':
        if 'vgg' not in config:
            raise KeyError(f"'vgg' section not found in config file at {config_path}.")
        # model = return_vgg(config['vgg'], device)
    return model

def return_densenet(config, device):
    num_blocks = int(config['num_blocks'])
    num_layers_per_block = int(config['num_layers_per_block'])
    growth_rate = int(config['growth_rate'])
    reduction = float(config['reduction'])
    return DenseNet(num_blocks, num_layers_per_block, growth_rate, reduction).to(device)

def return_resnet(config, device):
    # TODO: Implement this when the ResNet model is added
    pass

def return_vgg(config, device):
    # TODO: Implement this when the VGG model is added
    pass