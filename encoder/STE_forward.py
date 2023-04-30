import os
import csv
import sys
import torch

from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from core.dataset import EDATempDatasetAuxLosses2

from core.util import configure_backbone_forward_phase

import core.custom_transforms as ct
import core.config as exp_conf


# Written for simplicity, paralelize/shard as you wish
if __name__ == '__main__':
    clip_length = int(sys.argv[1])
    cuda_device_number = str(sys.argv[2])
    # Dont forget to assign this same size on ./core/custom_transforms
    image_size = (144, 144)

    io_config = exp_conf.STE_inputs
    
    model_weights = io_config['model_weights']
    target_directory = io_config['csv_save_directory']
    
    opt_config = exp_conf.STE_forward_params
    opt_config['batch_size'] = 1

    # cuda config
    backbone = configure_backbone_forward_phase(
        opt_config['backbone'], model_weights, clip_length)
    has_cuda = torch.cuda.is_available()
    device = torch.device('cuda:'+cuda_device_number if has_cuda else 'cpu')
    backbone = backbone.to(device)

    video_data_transforms = {
        'val': ct.video_val
    }

    pkl_list = list()

    with open(target_directory+'/results'+'.csv', mode='w') as vf:
        vf_writer = csv.writer(
            vf, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        d_train = EDATempDatasetAuxLosses2(
            audio_root=io_config['audio_root'], csv_root=io_config['csv_root'], clip_length=clip_length)

        dl_train = DataLoader(
            d_train, batch_size=opt_config['batch_size'], shuffle=False, num_workers=opt_config['threads'])

        for idx, dl in enumerate(dl_train):
            print(' \t Forward iter ', idx, '/', len(dl_train), end='\r')
            data, emotion, target_seg, target_ts = dl
            EDAt, TEMPt, audio_data = data

            TEMPt = TEMPt.to(device)
            EDAt = EDAt.to(device)
            audio_data = audio_data.to(device)
            emotion = emotion.to(device)

            with torch.set_grad_enabled(False):
                EDA_temp_audio_out, EDA_out, temp_out, audio_out, feats = backbone(
                    EDAt, TEMPt, audio_data)

                preds = torch.argmax(EDA_temp_audio_out,
                                     1).detach().cpu().numpy()
                gts = torch.argmax(emotion, 1).detach().cpu().numpy()
                emotion = emotion.detach().cpu().numpy()

                feats = feats.detach().cpu().numpy()[0]

                vf_writer.writerow(
                    [target_ts[0], target_seg[0], emotion[0], gts[0], preds[0], list(feats)])

                pkl_list.append([target_ts[0], target_seg[0],
                                emotion[0], gts[0], preds[0], list(feats)])
