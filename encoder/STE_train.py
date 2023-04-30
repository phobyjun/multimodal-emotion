import os
import sys
import torch

from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from core.dataset import EDATempDatasetAuxLosses, EDATempDatasetAuxLossesV

from core.optimization import optimize_av_losses
from core.optimization import _train_model_av_losses
from core.io import set_up_log_and_ws_out
from core.util import configure_backbone

import core.custom_transforms as ct
import core.config as exp_conf
# import core.config as exp_conf


if __name__ == '__main__':
    # experiment Reproducibility
    torch.manual_seed(11)
    torch.cuda.manual_seed(22)
    torch.backends.cudnn.deterministic = True

    clip_length = int(sys.argv[1])
    cuda_device_number = str(sys.argv[2])

    # check these 3 are in order, everythine else is kind of automated
    model_name = 'ste_encoder'
    io_config = exp_conf.STE_inputs
    
    opt_config = exp_conf.STE_optimization_params
    opt_config['batch_size'] = 128

    # io config
    log, target_models = set_up_log_and_ws_out(io_config['models_out'],
                                               opt_config, model_name)

    # cuda config
    backbone = configure_backbone(opt_config['backbone'], clip_length)
    has_cuda = torch.cuda.is_available()
    device = torch.device('cuda:'+cuda_device_number if has_cuda else 'cpu')
    backbone = backbone.to(device)

    # Optimization config
    criterion = opt_config['criterion']
    optimizer = opt_config['optimizer'](backbone.parameters(),
                                        lr=opt_config['learning_rate'])
    scheduler = lr_scheduler.StepLR(optimizer, step_size=opt_config['step_size'],
                                    gamma=opt_config['gamma'])

    video_data_transforms = {
        'train': ct.video_train,
        'val': ct.video_val
    }

    d_train = EDATempDatasetAuxLosses(
        audio_root=io_config['audio_root'], csv_root=io_config['csv_root'], clip_length=clip_length)

    d_val = EDATempDatasetAuxLossesV(
        audio_root=io_config['audio_root'], csv_root=io_config['csv_root'], clip_length=clip_length)

    dl_train = DataLoader(d_train, batch_size=opt_config['batch_size'],
                          shuffle=True, num_workers=opt_config['threads'])

    dl_val = DataLoader(d_val, batch_size=opt_config['batch_size'],
                        shuffle=True, num_workers=opt_config['threads'])

    model = optimize_av_losses(backbone, dl_train, dl_val, device,
                               criterion, optimizer, scheduler,
                               num_epochs=opt_config['epochs'],
                               models_out=target_models, log=log)
