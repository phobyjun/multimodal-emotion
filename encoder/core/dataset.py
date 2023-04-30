import os
import math
import glob
import time
import random
import torch

from PIL import Image
from torch.utils import data
from torchvision.transforms import RandomCrop

import numpy as np
import core.io as io
import core.clip_utils as cu
import multiprocessing as mp

import pandas as pd
import pickle
import tqdm
import ast


class EDATempDatasetAuxLosses(data.Dataset):
    def __init__(self, audio_root, csv_root, clip_length):
        super().__init__()

        self.audio_root = audio_root
        self.csv_root = csv_root

        self.clip_length = clip_length
        self.half_clip_length = math.floor(self.clip_length/2)

        EDA_df = pd.read_csv(f'{csv_root}/EDA_aug.csv')
        TEMP_df = pd.read_csv(f'{csv_root}/TEMP_aug.csv')
        gt_df = pd.read_csv(f'{csv_root}/gt_aug.csv')

        joined_df = pd.merge(EDA_df, TEMP_df, on=['script', 'ts'], how='inner')

        self.joined_df = joined_df
        self.gt_df = gt_df

        joined_df_list = set(joined_df['script'])
        gt_df_list = set(gt_df['script'])

        self.entity_list = sorted(list(joined_df_list & gt_df_list))

        self.filtered_df = self.joined_df[self.joined_df['script'].isin(
            self.entity_list)]

        self.target_entity = list()

        for sess in self.entity_list:
            if int(sess.split('_')[0][-2:]) <= 32:
                self.target_entity.append(sess)

        self.all_entity = dict()

        with open(f'{csv_root}/all_entity.pickle', 'rb') as fr:
            self.all_entity = pickle.load(fr)

    def __len__(self):
        # return len(self.entity_list)
        return len(self.target_entity)

    def __getitem__(self, index):

        # 'Sess32_script04_User064F_026'
        target_seg = self.target_entity[index]
        target_dict_list = self.all_entity[target_seg]

        mid_index = random.randint(0, len(target_dict_list)-1)
        left_idx = mid_index-self.half_clip_length

        left_side_addic = 0
        right_side_addic = 0

        if left_idx < 0:
            left_side_addic = 0 - left_idx
            left_idx = 0

        right_idx = mid_index+self.half_clip_length

        cliped_list = target_dict_list[left_idx:right_idx+1]

        EDA = [dict_item['EDA_value'] for dict_item in cliped_list]
        TEMP = [dict_item['TEMP_value'] for dict_item in cliped_list]

        right_side_addic = self.clip_length - left_side_addic - len(EDA)

        audio_offset = target_dict_list[0]['ts']

        audio_data = io.load_av_clip_from_metadata_TEST(
            cliped_list, self.audio_root, audio_offset, target_seg, self.clip_length)

        audio_data = np.float32(audio_data)

        for _ in range(left_side_addic):
            EDA.insert(0, EDA[0])
            TEMP.insert(0, TEMP[0])

        for _ in range(right_side_addic):
            EDA.insert(-1, EDA[-1])
            TEMP.insert(-1, TEMP[-1])

        EDAt = torch.tensor(EDA)
        TEMPt = torch.tensor(TEMP)

        emotion = target_dict_list[0]['emotion']

        emotion = torch.tensor(emotion)

        return (EDAt, TEMPt, audio_data), emotion


class EDATempDatasetAuxLossesV(data.Dataset):
    def __init__(self, audio_root, csv_root, clip_length):
        super().__init__()

        self.audio_root = audio_root
        self.csv_root = csv_root

        self.clip_length = clip_length
        self.half_clip_length = math.floor(self.clip_length/2)

        EDA_df = pd.read_csv(f'{csv_root}/EDA_aug.csv')
        TEMP_df = pd.read_csv(f'{csv_root}/TEMP_aug.csv')
        gt_df = pd.read_csv(f'{csv_root}/gt_aug.csv')

        joined_df = pd.merge(EDA_df, TEMP_df, on=['script', 'ts'], how='inner')

        self.joined_df = joined_df
        self.gt_df = gt_df

        joined_df_list = set(joined_df['script'])
        gt_df_list = set(gt_df['script'])

        self.entity_list = sorted(list(joined_df_list & gt_df_list))

        self.filtered_df = self.joined_df[self.joined_df['script'].isin(
            self.entity_list)]

        self.target_entity = list()

        for sess in self.entity_list:
            if int(sess.split('_')[0][-2:]) > 36:
                self.target_entity.append(sess)

        self.all_entity = dict()

        with open(f'{csv_root}/all_entity.pickle', 'rb') as fr:
            self.all_entity = pickle.load(fr)

    def __len__(self):

        return len(self.target_entity)

    def __getitem__(self, index):

        target_seg = self.target_entity[index]
        target_dict_list = self.all_entity[target_seg]

        mid_index = random.randint(0, len(target_dict_list)-1)
        left_idx = mid_index-self.half_clip_length

        left_side_addic = 0
        right_side_addic = 0

        if left_idx < 0:
            left_side_addic = 0 - left_idx
            left_idx = 0

        right_idx = mid_index+self.half_clip_length

        cliped_list = target_dict_list[left_idx:right_idx+1]

        EDA = [dict_item['EDA_value'] for dict_item in cliped_list]
        TEMP = [dict_item['TEMP_value'] for dict_item in cliped_list]

        right_side_addic = self.clip_length - left_side_addic - len(EDA)

        audio_offset = target_dict_list[0]['ts']

        audio_data = io.load_av_clip_from_metadata_TEST(
            cliped_list, self.audio_root, audio_offset, target_seg, self.clip_length)

        audio_data = np.float32(audio_data)

        for _ in range(left_side_addic):
            EDA.insert(0, EDA[0])
            TEMP.insert(0, TEMP[0])

        for _ in range(right_side_addic):
            EDA.insert(-1, EDA[-1])
            TEMP.insert(-1, TEMP[-1])

        EDAt = torch.tensor(EDA)
        TEMPt = torch.tensor(TEMP)

        emotion = target_dict_list[0]['emotion']
        emotion = torch.tensor(emotion)

        return (EDAt, TEMPt, audio_data), emotion


class EDATempDatasetAuxLosses2(data.Dataset):
    def __init__(self, audio_root, csv_root, clip_length):
        super().__init__()

        self.audio_root = audio_root
        self.csv_root = csv_root

        self.clip_length = clip_length
        self.half_clip_length = math.floor(self.clip_length/2)

        EDA_df = pd.read_csv(f'{csv_root}/EDA_aug.csv')
        TEMP_df = pd.read_csv(f'{csv_root}/TEMP_aug.csv')
        gt_df = pd.read_csv(f'{csv_root}/gt_aug.csv')

        joined_df = pd.merge(EDA_df, TEMP_df, on=['script', 'ts'], how='inner')

        self.joined_df = joined_df
        self.gt_df = gt_df

        joined_df_list = set(joined_df['script'])
        gt_df_list = set(gt_df['script'])

        self.entity_list = sorted(list(joined_df_list & gt_df_list))

        self.filtered_df = self.joined_df[self.joined_df['script'].isin(
            self.entity_list)]

        self.all_entity = dict()
        self.ts_entity = list()

        with open(f'{csv_root}/all_entity.pickle', 'rb') as fr:
            self.all_entity = pickle.load(fr)

        with open(f'{csv_root}/ts_script.pickle', 'rb') as fr:
            self.ts_entity = pickle.load(fr)

    def __len__(self):
        return len(self.ts_entity)

    def __getitem__(self, index):

        target_ts, target_seg = self.ts_entity[index]

        target_dict_list = self.all_entity[target_seg]

        fidx = 0
        for litem in target_dict_list:

            if litem['ts'] == target_ts:
                break

            fidx += 1

        mid_index = fidx
        left_idx = mid_index-self.half_clip_length

        left_side_addic = 0
        right_side_addic = 0

        if left_idx < 0:
            left_side_addic = 0 - left_idx
            left_idx = 0

        right_idx = mid_index+self.half_clip_length

        cliped_list = target_dict_list[left_idx:right_idx+1]

        EDA = [dict_item['EDA_value'] for dict_item in cliped_list]
        TEMP = [dict_item['TEMP_value'] for dict_item in cliped_list]

        right_side_addic = self.clip_length - left_side_addic - len(EDA)

        audio_offset = target_dict_list[0]['ts']

        audio_data = io.load_av_clip_from_metadata_TEST(
            cliped_list, self.audio_root, audio_offset, target_seg, self.clip_length)

        audio_data = np.float32(audio_data)

        for _ in range(left_side_addic):
            EDA.insert(0, EDA[0])
            TEMP.insert(0, TEMP[0])

        for _ in range(right_side_addic):
            EDA.insert(-1, EDA[-1])
            TEMP.insert(-1, TEMP[-1])

        EDAt = torch.tensor(EDA)
        TEMPt = torch.tensor(TEMP)

        emotion = target_dict_list[0]['emotion']

        emotion = torch.tensor(emotion)

        return (EDAt, TEMPt, audio_data), emotion, target_seg, target_ts
