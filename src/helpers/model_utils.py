import os
import configparser

# IMPORT MODELS HERE
from models.custom_compact_densenet import DenseNet
from models.custom_cnn import CustomCNN1
from models.resnet import ResNet, ResNetVersion
from models.resnet_scratch import get_resnet, MyResNetVersion, init_weights
from models.densenet169 import DenseNet169
from models.custom_cnn import CustomCNN1, BodyPartCNN, CustomCNNWithAttention
from models.vgg_pretrained import get_vgg_pretrained, VGGPretrainedVersion
from models.densenet121 import DenseNet121
from models.pretrained_densenets import PretrainedDenseNet, PretrainedDenseNetVersion

# from models.resnet import ResNet
from models.vgg import get_vgg, VGGVersion


def model_config_parser():
    config = configparser.ConfigParser()
    # Point to the main directory
    main_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    config_path = os.path.join(main_dir, "models.ini")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    config.read(config_path)
    return config, config_path


def get_hyperparameters():
    config, config_path = model_config_parser()
    if "hyperparams" not in config:
        raise KeyError(
            f"'hyperparameters' section not found in config file at {config_path}."
        )
    hyperarams = config["hyperparams"]
    lr = float(hyperarams["lr"])
    weight_decay = float(hyperarams["weight_decay"])
    num_epochs = int(hyperarams["num_epochs"])
    step_size = int(hyperarams["lr_scheduler_step_size"])
    gamma = float(hyperarams["lr_scheduler_gamma"])
    batch_size = int(hyperarams["batch_size"])
    factor = float(hyperarams["lr_scheduler_plateau_factor"])
    patience = int(hyperarams["lr_scheduler_plateau_patience"])
    return lr, weight_decay, num_epochs, step_size, gamma, batch_size, factor, patience


def get_model(model_name, device):
    config, config_path = model_config_parser()
    model = None
    if model_name == "custom_compact_densenet":
        if "custom_compact_densenet" not in config:
            raise KeyError(
                f"'custom_compact_densenet' section not found in config file at {config_path}."
            )
        model = return_custom_compact_densenet(
            config["custom_compact_densenet"], device
        )
    elif model_name == "densenet121":
        if "densenet121" not in config:
            raise KeyError(
                f"'densenet121' section not found in config file at {config_path}."
            )
        model = return_densenet121(config["densenet121"], device)
    elif model_name == "densenet169":
        if "densenet169" not in config:
            raise KeyError(
                f"'densenet169' section not found in config file at {config_path}."
            )
        model = return_densenet169(config["densenet169"], device)
    elif model_name == "pretrained_densenet169":
        if "pretrained_densenet169" not in config:
            raise KeyError(
                f"'pretrained_densenet169' section not found in config file at {config_path}."
            )
        model = return_pretrained_densenet(config["pretrained_densenet169"], device)
    elif model_name == "pretrained_densenet121":
        if "pretrained_densenet121" not in config:
            raise KeyError(
                f"'pretrained_densenet121' section not found in config file at {config_path}."
            )
        model = return_pretrained_densenet(config["pretrained_densenet121"], device)
    elif model_name == "resnet":
        if "resnet" not in config:
            raise KeyError(
                f"'resnet' section not found in config file at {config_path}."
            )
        model = return_resnet(config["resnet"], device)
    elif model_name == "resnet_scratch":
        if "resnet_scratch" not in config:
            raise KeyError(
                f"'resnet scratch' section not found in config file at {config_path}."
            )
        model = return_resnet_scratch(config["resnet_scratch"], device)
    elif model_name == "vgg":
        if "vgg" not in config:
            raise KeyError(f"'vgg' section not found in config file at {config_path}.")
        model = return_vgg(config["vgg"], device)
    elif model_name == "vgg_pretrained":
        if "vgg_pretrained" not in config:
            raise KeyError(
                f"'vgg_pretrained' section not found in config file at {config_path}."
            )
        model = return_vgg_pretrained(config["vgg_pretrained"], device)
    elif model_name == "custom_cnn1":
        if "custom_cnn1" not in config:
            raise KeyError(
                f"'custom_cnn1' section not found in config file at {config_path}."
            )
        model = return_custom_cnn1(config["custom_cnn1"], device)
    elif model_name == "body_part_cnn":
        if "body_part_cnn" not in config:
            raise KeyError(
                f"'body_part_cnn' section not found in config file at {config_path}."
            )
        model = return_body_part_cnn(config["body_part_cnn"], device)
    elif model_name == "custom_cnn_attention":
        if "custom_cnn_attention" not in config:
            raise KeyError(
                f"'custom_cnn_attention' section not found in config file at {config_path}."
            )
        model = return_custom_cnn_attention(config["custom_cnn_attention"], device)
    return model


def return_custom_compact_densenet(config, device):
    num_blocks = int(config["num_blocks"])
    num_layers_per_block = int(config["num_layers_per_block"])
    growth_rate = int(config["growth_rate"])
    reduction = float(config["reduction"])
    return DenseNet(num_blocks, num_layers_per_block, growth_rate, reduction).to(device)


def return_densenet121(config, device):
    num_classes = int(config["num_classes"])
    return DenseNet121(num_classes).to(device)


def return_densenet169(config, device):
    num_classes = int(config["num_classes"])
    return DenseNet169(num_classes).to(device)


def return_resnet(config, device):
    # TODO: Implement this when the ResNet model is added
    num_labels = int(config["num_labels"])
    pretrained = bool(config["pretrained"])
    variant = str(config["variant"])
    try:
        resnet_version = ResNetVersion(variant)
        return ResNet(num_labels, pretrained=pretrained, variant=resnet_version).to(
            device
        )
    except ValueError:
        print(f"Invalid variant: {variant}. Add to ResNet Version enum")

    return None


def return_resnet_scratch(config, device):
    num_labels = int(config["num_labels"])
    variant = str(config["variant"])
    try:
        resnet_version = MyResNetVersion(variant)
        model = get_resnet(num_labels, resnet_version)
        if model:
            init_weights(model)
            return model.to(device)
    except ValueError:
        print(f"Invalid variant: {variant}. Add to ResNet Version enum")

    return None


def return_vgg(config, device):
    # TODO: Implement this when the VGG model is added
    # pass
    num_classes = int(config["num_classes"])
    model = config["model"]
    try:
        resnet_version = VGGVersion(model)
        model = get_vgg(num_classes, resnet_version)
        if model:
            return model.to(device)
    except ValueError:
        print(f"Invalid variant: {model}. Add to VGG Version enum")

    return None

def return_vgg_pretrained(config, device):
    num_classes = int(config["num_classes"])
    pretrained = config["pretrained"]
    model = config["model"]
    try:
        vgg_version = VGGPretrainedVersion(model)
        model = get_vgg_pretrained(num_classes, vgg_version, pretrained)
        if model:
            return model.to(device)
    except ValueError:
        print(f"Invalid variant: {model}. Add to VGG Version enum")

    return None



def return_custom_cnn1(config, device):
    dropout_rate = float(config["dropout_rate"])
    num_classes = int(config["num_classes"])
    return CustomCNN1(dropout_rate=dropout_rate, num_classes=num_classes).to(device)


def return_body_part_cnn(config, device):
    return BodyPartCNN().to(device)


def return_custom_cnn_attention(config, device):
    dropout_rate = float(config["dropout_rate"])
    num_classes = int(config["num_classes"])
    return CustomCNNWithAttention(
        dropout_rate=dropout_rate, num_classes=num_classes
    ).to(device)


def return_pretrained_densenet(config, device):
    num_classes = int(config["num_classes"])
    pretrained = bool(config["pretrained"])
    variant = str(config["variant"])
    try:
        pretrained_densenet_version = PretrainedDenseNetVersion(variant)
        return PretrainedDenseNet(
            num_classes, pretrained=pretrained, variant=pretrained_densenet_version
        ).to(device)
    except ValueError:
        print(f"Invalid variant: {variant}. Add to pre-trained DenseNet Version enum")

    return None
