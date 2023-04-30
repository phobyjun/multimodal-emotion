import sys

import torch.nn as nn
import torch.optim as optim
import core.models as mdet

STE_inputs = {
    'models_out': './STE_TRAIN',

    'csv_root': './KEMDy20',
    'audio_root': './KEMDy20/wav',

    'model_weights': './STE_TRAIN/ste_encoder/1.pth',
    'csv_save_directory': './KEMDy20',
}

# Optimization params
STE_optimization_params = {
    # Net Arch
    'backbone': mdet.resnet18_two_streams_TEST,

    # Optimization config
    'optimizer': optim.Adam,
    'criterion': nn.CrossEntropyLoss(),
    'learning_rate': 3e-4,
    'epochs': 100,
    'step_size': 40,
    'gamma': 0.1,

    # Batch Config
    'batch_size': 128,
    'threads': 1
}

STE_forward_params = {
    # Net Arch
    'backbone': mdet.resnet18_two_streams_forward_TEST,

    # Batch Config
    'batch_size': 1,
    'threads': 1
}
