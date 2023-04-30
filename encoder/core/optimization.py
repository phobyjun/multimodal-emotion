import os
import time
import copy
import torch

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score

import numpy as np

def optimize_av_losses(model, dataloader_train, data_loader_val, device,
                            criterion, optimizer, scheduler, num_epochs,
                            models_out=None, log=None):

    for epoch in range(num_epochs):
        print()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        outs_train = _train_model_av_losses(model, dataloader_train, optimizer,
                                                 criterion, scheduler, device)
        outs_val = _test_model_av_losses(model, data_loader_val, optimizer,
                                              criterion, scheduler, device)

        train_loss, train_loss_a, train_loss_v, _, _ = outs_train

        val_loss, val_loss_a, val_loss_v, _, _ = outs_val

        if models_out is not None:
            model_target = os.path.join(models_out, str(epoch+1)+'.pth')
            print('save model to ', model_target)
            torch.save(model.state_dict(), model_target)

        if log is not None:
            log.writeDataLog([epoch+1, train_loss, val_loss, ])

    return model

def _train_model_av_losses(model, dataloader, optimizer, criterion,
                                scheduler, device):
    softmax_layer = torch.nn.Softmax(dim=1)
    model.train()
    pred_lst = []
    label_lst = []

    running_loss_ETA = 0.0
    running_loss_E = 0.0
    running_loss_T = 0.0
    running_loss_A = 0.0
    running_corrects = 0

    test_num = 0

    # Iterate over data
    for idx, dl in enumerate(dataloader):
        print('\t Train iter ', idx, '/', len(dataloader), end='\r')

        data, emotion = dl

        EDAt, TEMPt, audio_data = data

        TEMPt = TEMPt.to(device)
        EDAt = EDAt.to(device)
        audio_data = audio_data.to(device)

        emotion = emotion.to(device)

        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            
            EDA_temp_audio_out, EDA_out, temp_out, audio_out, _ = model(
                EDAt, TEMPt, audio_data)
            
            preds = torch.argmax(EDA_temp_audio_out, 1)

            emotion = emotion.to(torch.float32)

            loss_ETA = criterion(EDA_temp_audio_out, emotion)
            loss_E = criterion(EDA_out, emotion)
            loss_T = criterion(temp_out, emotion)
            loss_A = criterion(audio_out, emotion)
            loss = loss_ETA + loss_E + loss_T + loss_A

            gts = torch.argmax(emotion, 1)

            loss.backward()
            optimizer.step()

        with torch.set_grad_enabled(False):
            label_lst.extend(torch.argmax(emotion, 1).cpu().numpy())
            pred_lst.extend(torch.argmax(EDA_temp_audio_out, 1).cpu().numpy())

        # statistics
        running_loss_ETA += loss_ETA.item()
        running_loss_E += loss_E.item()
        running_loss_T += loss_T.item()
        running_loss_A += loss_A.item()
        running_corrects += torch.sum(preds == gts)

        test_num += preds.shape[0]

    scheduler.step()

    epoch_loss_ETA = running_loss_ETA / test_num
    epoch_loss_E = running_loss_E / test_num
    epoch_loss_T = running_loss_T / test_num
    epoch_loss_A = running_loss_A / test_num
    epoch_acc = running_corrects.double() / test_num

    label_lst = np.array(label_lst)
    pred_lst = np.array(pred_lst)

    f1 = f1_score(label_lst, pred_lst, average='micro')

    print('ETA Loss: {:.4f} E Loss: {:.4f} T Loss: {:.4f} A Loss: {:.4f}  ACC: {:.4f} f1: {:.4f}'.format(
          epoch_loss_ETA, epoch_loss_E, epoch_loss_T, epoch_loss_A, epoch_acc, f1))
    return epoch_loss_ETA, epoch_loss_E, epoch_loss_T, epoch_loss_A, epoch_acc

def _test_model_av_losses(model, dataloader, optimizer, criterion, scheduler, device):
    softmax_layer = torch.nn.Softmax(dim=1)
    model.train()
    pred_lst = []
    label_lst = []

    running_loss_ETA = 0.0
    running_loss_E = 0.0
    running_loss_T = 0.0
    running_loss_A = 0.0
    running_corrects = 0

    test_num = 0

    # Iterate over data
    for idx, dl in enumerate(dataloader):
        print('\t Train iter ', idx, '/', len(dataloader), end='\r')

        data, emotion = dl

        EDAt, TEMPt, audio_data = data

        TEMPt = TEMPt.to(device)
        EDAt = EDAt.to(device)
        audio_data = audio_data.to(device)

        emotion = emotion.to(device)

        optimizer.zero_grad()
        with torch.set_grad_enabled(False):
            
            EDA_temp_audio_out, EDA_out, temp_out, audio_out, _ = model(
                EDAt, TEMPt, audio_data)
            
            preds = torch.argmax(EDA_temp_audio_out, 1)
            

            emotion = emotion.to(torch.float32)

            loss_ETA = criterion(EDA_temp_audio_out, emotion)
            loss_E = criterion(EDA_out, emotion)
            loss_T = criterion(temp_out, emotion)
            loss_A = criterion(audio_out, emotion)
            loss = loss_ETA + loss_E + loss_T + loss_A

            gts = torch.argmax(emotion, 1)

        with torch.set_grad_enabled(False):
            label_lst.extend(torch.argmax(emotion, 1).cpu().numpy())
            pred_lst.extend(torch.argmax(EDA_temp_audio_out, 1).cpu().numpy())

        # statistics
        running_loss_ETA += loss_ETA.item()
        running_loss_E += loss_E.item()
        running_loss_T += loss_T.item()
        running_loss_A += loss_A.item()
        running_corrects += torch.sum(preds == gts)

        test_num += preds.shape[0]

    epoch_loss_ETA = running_loss_ETA / test_num
    epoch_loss_E = running_loss_E / test_num
    epoch_loss_T = running_loss_T / test_num
    epoch_loss_A = running_loss_A / test_num
    epoch_acc = running_corrects.double() / test_num

    label_lst = np.array(label_lst)
    pred_lst = np.array(pred_lst)
    
    f1 = f1_score(label_lst, pred_lst, average='micro')
    
    print('ETA Loss: {:.4f} E Loss: {:.4f} T Loss: {:.4f} A Loss: {:.4f}  ACC: {:.4f} f1: {:.4f}'.format(
          epoch_loss_ETA, epoch_loss_E, epoch_loss_T, epoch_loss_A, epoch_acc, f1))
    return epoch_loss_ETA, epoch_loss_E, epoch_loss_T, epoch_loss_A, epoch_acc