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
import tqdm
import ast


class CachedAVSource(data.Dataset):
    def __init__(self):
        # Cached data
        self.entity_data = {}
        self.speech_data = {}
        self.entity_list = []

        # Reproducibilty
        random.seed(42)
        np.random.seed(0)

    def _postprocess_speech_label(self, speech_label):
        speech_label = int(speech_label)
        if speech_label == 2:  # Remember 2 = SPEAKING_NOT_AUDIBLE
            speech_label = 0
        return speech_label

    def _postprocess_entity_label(self, entity_label):
        entity_label = int(entity_label)
        if entity_label == 2:  # Remember 2 = SPEAKING_NOT_AUDIBLE
            entity_label = 0
        return entity_label

    def _cache_entity_data(self, csv_file_path):
        entity_set = set()

        csv_data = io.csv_to_list(csv_file_path)
        csv_data.pop(0)  # CSV header
        for csv_row in csv_data:
            video_id = csv_row[0]
            entity_id = csv_row[-3]
            timestamp = csv_row[1]

            speech_label = self._postprocess_speech_label(csv_row[-2])
            entity_label = self._postprocess_entity_label(csv_row[-2])
            minimal_entity_data = (entity_id, timestamp, entity_label)

            # Store minimal entity data
            if video_id not in self.entity_data.keys():
                self.entity_data[video_id] = {}
            if entity_id not in self.entity_data[video_id].keys():
                self.entity_data[video_id][entity_id] = []
                entity_set.add((video_id, entity_id))
            self.entity_data[video_id][entity_id].append(minimal_entity_data)

            # Store speech meta-data
            if video_id not in self.speech_data.keys():
                self.speech_data[video_id] = {}
            if timestamp not in self.speech_data[video_id].keys():
                self.speech_data[video_id][timestamp] = speech_label

            # max operation yields if someone is speaking.
            new_speech_label = max(
                self.speech_data[video_id][timestamp], speech_label)
            self.speech_data[video_id][timestamp] = new_speech_label

        return entity_set

    def _cache_entity_data_forward(self, csv_file_path, target_video):
        entity_list = list()

        csv_data = io.csv_to_list(csv_file_path)
        csv_data.pop(0)  # CSV header
        for csv_row in csv_data:
            video_id = csv_row[0]  # -FaXLcSFjUI
            if video_id != target_video:
                continue

            entity_id = csv_row[-3]  # -FaXLcSFjUI_1260_1320:39
            timestamp = csv_row[1]  # 1298.9, 1298.93
            entity_label = self._postprocess_entity_label(
                csv_row[-2])  # 0[not speaking] or 1[speaking]

            # (-FaXLcSFjUI, FaXLcSFjUI_1260_1320:39, 1298.9)
            entity_list.append((video_id, entity_id, timestamp))
            # sfate to ingore label here
            minimal_entity_data = (entity_id, timestamp, entity_label)

            if video_id not in self.entity_data.keys():
                self.entity_data[video_id] = {}

            if entity_id not in self.entity_data[video_id].keys():
                self.entity_data[video_id][entity_id] = []
            self.entity_data[video_id][entity_id].append(minimal_entity_data)

        return entity_list

    def _entity_list_postprocessing(self, entity_set):
        print('Initial', len(entity_set))

        # filter out missing data on disk
        all_disk_data = set(os.listdir(self.video_root))
        for video_id, entity_id in entity_set.copy():
            if entity_id not in all_disk_data:
                entity_set.remove((video_id, entity_id))
        print('Pruned not in disk', len(entity_set))
        self.entity_list = sorted(list(entity_set))


class AudioVideoDatasetAuxLosses(CachedAVSource):
    def __init__(self, audio_root, video_root, csv_file_path, clip_lenght,
                 target_size, video_transform=None, do_video_augment=False):
        super().__init__()

        # Data directories
        self.audio_root = audio_root
        self.video_root = video_root

        # Post-processing
        self.video_transform = video_transform
        self.do_video_augment = do_video_augment

        # Clip arguments
        self.clip_lenght = clip_lenght
        self.half_clip_length = math.floor(self.clip_lenght/2)
        self.target_size = target_size

        entity_set = self._cache_entity_data(csv_file_path)
        self._entity_list_postprocessing(entity_set)

    def __len__(self):
        return int(len(self.entity_list)/1)

    def __getitem__(self, index):
        # Get meta-data
        video_id, entity_id = self.entity_list[index]
        # print("entity_list: ", self.entity_list)

        # entity_list:  [('-FaXLcSFjUI', '-FaXLcSFjUI_0900_0960:12'), ('-FaXLcSFjUI', '-FaXLcSFjUI_0900_0960:14'), ('-FaXLcSFjUI', '-FaXLcSFjUI_0900_0960:22'), ('-FaXLcSFjUI', '-FaXLcSFjUI_0900_0960:24'), ('-FaXLcSFjUI', '-FaXLcSFjUI_0900_0960:3'), ('-FaXLcSFjUI', '-FaXLcSFjUI_0900_0960:6'), ('-FaXLcSFjUI', '-FaXLcSFjUI_0900_0960:7'), ('-FaXLcSFjUI', '-FaXLcSFjUI_0900_0960:8'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1080_1140:11'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1080_1140:12'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1080_1140:17'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1080_1140:19'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1080_1140:21'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1080_1140:22'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1080_1140:24'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1080_1140:7'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1140_1200:1'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1140_1200:10'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1140_1200:11'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1140_1200:18'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1140_1200:20'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1140_1200:5'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1140_1200:7'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1140_1200:9'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1200_1260:15'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1200_1260:21'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1200_1260:46'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1260_1320:39'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1260_1320:47'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1320_1380:1'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1320_1380:12'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1320_1380:26'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1320_1380:46'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1320_1380:52'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1320_1380:8'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1380_1440:31'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1380_1440:34'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1380_1440:36'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1380_1440:37'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1380_1440:41'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1380_1440:42'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1380_1440:60'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1440_1500:22'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1440_1500:28'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1440_1500:30'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1440_1500:40'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1440_1500:42'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1500_1560:26'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1500_1560:31'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1500_1560:51'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1560_1620:13'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1560_1620:17'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1560_1620:2'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1560_1620:23'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1560_1620:3'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1560_1620:4'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1560_1620:41'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1560_1620:47'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1560_1620:52'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1560_1620:57'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1560_1620:59'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1620_1680:15'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1620_1680:2'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1620_1680:30'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1680_1740:22'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1680_1740:25'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1680_1740:28'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1680_1740:5'), ('-FaXLcSFjUI','-FaXLcSFjUI_1740_1800:12'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1740_1800:13'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1740_1800:14'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1740_1800:15'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1740_1800:23'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1740_1800:24')]

        # print("index, video_id, entity_id:\n", index, video_id, entity_id)
        entity_metadata = self.entity_data[video_id][entity_id]

        audio_offset = float(entity_metadata[0][1])
        mid_index = random.randint(0, len(entity_metadata)-1)
        midone = entity_metadata[mid_index]
        target = int(midone[-1])
        target_audio = self.speech_data[video_id][midone[1]]
        # print("self.speech_data: ", self.speech_data)
        # self.speech_data[video_id] => {timestamp: label} and if there is speech at that time then, label = 1
        # self.speech_data:
        # self.speech_data:  {'-FaXLcSFjUI': {'1298.9': 0, '1298.93': 0, '1298.97': 0, '1299.0': 0, '1299.03':0, '1299.07': 0, '1299.1': 0, '1299.13': 0, '1299.17': 0, '1299.2': 0, '1299.23': 0, '1299.27': 0, '1299.3': 0, '1299.33': 0, '1299.37': 0, '1299.4': 0, '1299.43': 0, '1299.47': 0, '1299.5': 0, '1299.53': 0, '1299.57': 0, '129
        # For checking at that timestamp, there is a person speaking or not

        # print('audio_offset, mid_index, midone, target, target_audio:',
        #       audio_offset, mid_index, midone, target, target_audio)

        # print("video_id, entity_id: ", video_id, entity_id)
        # print("entity_metadata: ", entity_metadata)
        # print('audio_offset: ', audio_offset)
        # print('mid_index: ', mid_index)
        # print('midone: ', midone)
        # print('target: ', target)
        # print('target_audio: ', target_audio)

        # video_id, entity_id:  -FaXLcSFjUI -FaXLcSFjUI_1560_1620:4
        # entity_metadata:  [('-FaXLcSFjUI_1560_1620:4', '1583.89', 1), ('-FaXLcSFjUI_1560_1620:4', '1583.92',1), ('-FaXLcSFjUI_1560_1620:4', '1583.95', 1), ('-FaXLcSFjUI_1560_1620:4', '1583.99', 1), ('-FaXLcSFjUI_1560_1620:4', '1584.02', 1), ('-FaXLcSFjUI_1560_1620:4', '1584.05', 1), ('-FaXLcSFjUI_1560_1620:4', '1584.09', 1), ('-FaXLcSFjUI_1560_1620:4', '1584.12', 1), ('-FaXLcSFjUI_1560_1620:4', '1584.15', 1), ('-FaXLcSFjUI_1560_1620:4', '1584.19', 1), ('-FaXLcSFjUI_1560_1620:4', '1584.22', 1), ('-FaXLcSFjUI_1560_1620:4', '1584.25', 1), ('-FaXLcSFjUI_1560_1620:4', '1584.29', 1), ('-FaXLcSFjUI_1560_1620:4','1584.32', 1), ('-FaXLcSFjUI_1560_1620:4', '1584.35', 1), ('-FaXLcSFjUI_1560_1620:4', '1584.39', 1),('-FaXLcSFjUI_1560_1620:4', '1584.42', 0), ('-FaXLcSFjUI_1560_1620:4', '1584.45', 0), ('-FaXLcSFjUI_1560_1620:4', '1584.49', 0), ('-FaXLcSFjUI_1560_1620:4', '1584.52', 0), ('-FaXLcSFjUI_1560_1620:4', '1584.55', 0), ('-FaXLcSFjUI_1560_1620:4', '1584.59', 0), ('-FaXLcSFjUI_1560_1620:4', '1584.62', 0), ('-FaXLcSFjUI_1560_1620:4', '1584.65', 0), ('-FaXLcSFjUI_1560_1620:4', '1584.69', 0), ('-FaXLcSFjUI_1560_1620:4', '1584.72', 0), ('-FaXLcSFjUI_1560_1620:4', '1584.75', 0), ('-FaXLcSFjUI_1560_1620:4', '1584.79', 0), ('-FaXLcSFjUI_1560_1620:4', '1584.82', 0), ('-FaXLcSFjUI_1560_1620:4', '1584.85', 0), ('-FaXLcSFjUI_1560_1620:4', '1584.89', 0), ('-FaXLcSFjUI_1560_1620:4', '1584.92', 0), ('-FaXLcSFjUI_1560_1620:4', '1584.95', 0), ('-FaXLcSFjUI_1560_1620:4', '1584.99', 0)]
        # audio_offset:  1583.89
        # mid_index:  6
        # midone:  ('-FaXLcSFjUI_1560_1620:4', '1584.09', 1)
        # target:  1
        # target_audio:  1

        clip_meta_data = cu.generate_clip_meta(entity_metadata, mid_index,
                                               self.half_clip_length)

        # clip meta data
        # [('-FaXLcSFjUI_1560_1620:17', '1596.33', 0),
        # ('-FaXLcSFjUI_1560_1620:17', '1596.36', 0),
        # ('-FaXLcSFjUI_1560_1620:17', '1596.4', 0),
        # ('-FaXLcSFjUI_1560_1620:17', '1596.43', 0),
        # ('-FaXLcSFjUI_1560_1620:17', '1596.46', 0),
        # ('-FaXLcSFjUI_1560_1620:17', '1596.5', 0),
        # ('-FaXLcSFjUI_1560_1620:17', '1596.53', 0),
        # ('-FaXLcSFjUI_1560_1620:17', '1596.56', 0),
        # ('-FaXLcSFjUI_1560_1620:17', '1596.6', 0),
        # ('-FaXLcSFjUI_1560_1620:17', '1596.63', 0),
        # ('-FaXLcSFjUI_1560_1620:17', '1596.67', 0)]

        video_data, audio_data = io.load_av_clip_from_metadata(clip_meta_data,
                                                               self.video_root, self.audio_root, audio_offset,
                                                               self.target_size)

        # self.target_size 144

        if self.do_video_augment:
            # random flip
            if bool(random.getrandbits(1)):
                video_data = [s.transpose(Image.FLIP_LEFT_RIGHT)
                              for s in video_data]

            # random crop
            width, height = video_data[0].size
            f = random.uniform(0.5, 1)
            i, j, h, w = RandomCrop.get_params(
                video_data[0], output_size=(int(height*f), int(width*f)))
            video_data = [s.crop(box=(j, i, w, h)) for s in video_data]

        if self.video_transform is not None:
            video_data = [self.video_transform(vd) for vd in video_data]

        video_data = torch.cat(video_data, dim=0)
        return (np.float32(audio_data), video_data), target, target_audio


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

        self.all_entity = dict()

        for target_script in tqdm.tqdm(self.entity_list):
            target_script_df = self.filtered_df[self.filtered_df['script']
                                                == target_script]

            for index, row in target_script_df.iterrows():
                new_entity = {
                    'ts': row.ts,
                    'EDA_value': row.EDA_value,
                    'TEMP_value': row.TEMP_value,
                    'emotion': self.gt_df[self.gt_df['script'] == target_script]['emotion_one_hot'].item()
                }

                if target_script not in self.all_entity.keys():
                    self.all_entity[target_script] = list()
                    self.all_entity[target_script].append(new_entity)
                else:
                    self.all_entity[target_script].append(new_entity)

        print("self.all_entity: ", self.all_entity)

    def __len__(self):
        return len(self.entity_list)

    def __getitem__(self, index):

        # 'Sess32_script04_User064F_026'
        target_seg = self.entity_list[index]
        session, script, user, sequence = self.entity_list[index].split('_')

        target_df = self.joined_df[self.joined_df['script'] == target_seg]

        # print("target_df: ", target_df)
        # print("target_seg: ", target_seg)
        # print("0, len(target_df)-1: ", 0, len(target_df)-1)

        mid_index = random.randint(0, len(target_df)-1)
        start_idx = target_df.index[0]
        left_idx = mid_index-self.half_clip_length

        left_side_addic = 0
        right_side_addic = 0

        if left_idx < 0:
            left_side_addic = 0 - left_idx
            left_idx = 0

        right_idx = mid_index+self.half_clip_length

        cliped_df = target_df.iloc[left_idx:right_idx, :]

        EDA = cliped_df['EDA_value']
        TEMP = cliped_df['TEMP_value']

        right_side_addic = self.clip_length - left_side_addic - 1 - len(EDA)

        EDA = EDA.values.tolist()
        TEMP = TEMP.values.tolist()

        for _ in range(left_side_addic):
            EDA.insert(0, EDA[0])
            TEMP.insert(0, TEMP[0])

        for _ in range(right_side_addic):
            EDA.insert(-1, EDA[-1])
            TEMP.insert(-1, TEMP[-1])

        EDAt = torch.tensor(EDA)
        TEMPt = torch.tensor(TEMP)

        emotion = self.gt_df[self.gt_df['script']
                             == target_seg]['emotion_one_hot'].item()

        emotion = ast.literal_eval(emotion)
        emotion = torch.tensor(emotion)

        # print("type(emotion): ", type(emotion))

        # print("emotion: ", emotion)
        # print("type(emotion): ", type(emotion))

        return (EDAt, TEMPt), emotion


# class EDATempDatasetAuxLosses(data.Dataset):
#     def __init__(self, audio_root, csv_file_root, gt_csv, clip_lenght):
#         super().__init__()

#         # Data directories
#         self.audio_root = audio_root
#         self.video_root = video_root

#         # Post-processing
#         self.video_transform = video_transform
#         self.do_video_augment = do_video_augment

#         # Clip arguments
#         self.clip_lenght = clip_lenght
#         self.half_clip_length = math.floor(self.clip_lenght/2)
#         self.target_size = target_size

#         entity_set = self._cache_entity_data(csv_file_path)
#         self._entity_list_postprocessing(entity_set)

#     def __len__(self):
#         return int(len(self.entity_list)/1)

#     def __getitem__(self, index):
#         # Get meta-data
#         video_id, entity_id = self.entity_list[index]
#         # print("entity_list: ", self.entity_list)

#         # entity_list:  [('-FaXLcSFjUI', '-FaXLcSFjUI_0900_0960:12'), ('-FaXLcSFjUI', '-FaXLcSFjUI_0900_0960:14'), ('-FaXLcSFjUI', '-FaXLcSFjUI_0900_0960:22'), ('-FaXLcSFjUI', '-FaXLcSFjUI_0900_0960:24'), ('-FaXLcSFjUI', '-FaXLcSFjUI_0900_0960:3'), ('-FaXLcSFjUI', '-FaXLcSFjUI_0900_0960:6'), ('-FaXLcSFjUI', '-FaXLcSFjUI_0900_0960:7'), ('-FaXLcSFjUI', '-FaXLcSFjUI_0900_0960:8'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1080_1140:11'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1080_1140:12'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1080_1140:17'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1080_1140:19'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1080_1140:21'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1080_1140:22'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1080_1140:24'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1080_1140:7'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1140_1200:1'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1140_1200:10'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1140_1200:11'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1140_1200:18'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1140_1200:20'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1140_1200:5'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1140_1200:7'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1140_1200:9'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1200_1260:15'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1200_1260:21'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1200_1260:46'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1260_1320:39'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1260_1320:47'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1320_1380:1'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1320_1380:12'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1320_1380:26'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1320_1380:46'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1320_1380:52'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1320_1380:8'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1380_1440:31'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1380_1440:34'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1380_1440:36'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1380_1440:37'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1380_1440:41'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1380_1440:42'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1380_1440:60'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1440_1500:22'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1440_1500:28'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1440_1500:30'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1440_1500:40'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1440_1500:42'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1500_1560:26'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1500_1560:31'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1500_1560:51'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1560_1620:13'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1560_1620:17'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1560_1620:2'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1560_1620:23'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1560_1620:3'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1560_1620:4'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1560_1620:41'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1560_1620:47'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1560_1620:52'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1560_1620:57'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1560_1620:59'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1620_1680:15'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1620_1680:2'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1620_1680:30'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1680_1740:22'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1680_1740:25'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1680_1740:28'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1680_1740:5'), ('-FaXLcSFjUI','-FaXLcSFjUI_1740_1800:12'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1740_1800:13'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1740_1800:14'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1740_1800:15'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1740_1800:23'), ('-FaXLcSFjUI', '-FaXLcSFjUI_1740_1800:24')]

#         # print("index, video_id, entity_id:\n", index, video_id, entity_id)
#         entity_metadata = self.entity_data[video_id][entity_id]

#         audio_offset = float(entity_metadata[0][1])
#         mid_index = random.randint(0, len(entity_metadata)-1)
#         midone = entity_metadata[mid_index]
#         target = int(midone[-1])
#         target_audio = self.speech_data[video_id][midone[1]]
#         # print("self.speech_data: ", self.speech_data)
#         # self.speech_data[video_id] => {timestamp: label} and if there is speech at that time then, label = 1
#         # self.speech_data:
#         # self.speech_data:  {'-FaXLcSFjUI': {'1298.9': 0, '1298.93': 0, '1298.97': 0, '1299.0': 0, '1299.03':0, '1299.07': 0, '1299.1': 0, '1299.13': 0, '1299.17': 0, '1299.2': 0, '1299.23': 0, '1299.27': 0, '1299.3': 0, '1299.33': 0, '1299.37': 0, '1299.4': 0, '1299.43': 0, '1299.47': 0, '1299.5': 0, '1299.53': 0, '1299.57': 0, '129
#         # For checking at that timestamp, there is a person speaking or not

#         # print('audio_offset, mid_index, midone, target, target_audio:',
#         #       audio_offset, mid_index, midone, target, target_audio)

#         # print("video_id, entity_id: ", video_id, entity_id)
#         # print("entity_metadata: ", entity_metadata)
#         # print('audio_offset: ', audio_offset)
#         # print('mid_index: ', mid_index)
#         # print('midone: ', midone)
#         # print('target: ', target)
#         # print('target_audio: ', target_audio)

#         # video_id, entity_id:  -FaXLcSFjUI -FaXLcSFjUI_1560_1620:4
#         # entity_metadata:  [('-FaXLcSFjUI_1560_1620:4', '1583.89', 1), ('-FaXLcSFjUI_1560_1620:4', '1583.92',1), ('-FaXLcSFjUI_1560_1620:4', '1583.95', 1), ('-FaXLcSFjUI_1560_1620:4', '1583.99', 1), ('-FaXLcSFjUI_1560_1620:4', '1584.02', 1), ('-FaXLcSFjUI_1560_1620:4', '1584.05', 1), ('-FaXLcSFjUI_1560_1620:4', '1584.09', 1), ('-FaXLcSFjUI_1560_1620:4', '1584.12', 1), ('-FaXLcSFjUI_1560_1620:4', '1584.15', 1), ('-FaXLcSFjUI_1560_1620:4', '1584.19', 1), ('-FaXLcSFjUI_1560_1620:4', '1584.22', 1), ('-FaXLcSFjUI_1560_1620:4', '1584.25', 1), ('-FaXLcSFjUI_1560_1620:4', '1584.29', 1), ('-FaXLcSFjUI_1560_1620:4','1584.32', 1), ('-FaXLcSFjUI_1560_1620:4', '1584.35', 1), ('-FaXLcSFjUI_1560_1620:4', '1584.39', 1),('-FaXLcSFjUI_1560_1620:4', '1584.42', 0), ('-FaXLcSFjUI_1560_1620:4', '1584.45', 0), ('-FaXLcSFjUI_1560_1620:4', '1584.49', 0), ('-FaXLcSFjUI_1560_1620:4', '1584.52', 0), ('-FaXLcSFjUI_1560_1620:4', '1584.55', 0), ('-FaXLcSFjUI_1560_1620:4', '1584.59', 0), ('-FaXLcSFjUI_1560_1620:4', '1584.62', 0), ('-FaXLcSFjUI_1560_1620:4', '1584.65', 0), ('-FaXLcSFjUI_1560_1620:4', '1584.69', 0), ('-FaXLcSFjUI_1560_1620:4', '1584.72', 0), ('-FaXLcSFjUI_1560_1620:4', '1584.75', 0), ('-FaXLcSFjUI_1560_1620:4', '1584.79', 0), ('-FaXLcSFjUI_1560_1620:4', '1584.82', 0), ('-FaXLcSFjUI_1560_1620:4', '1584.85', 0), ('-FaXLcSFjUI_1560_1620:4', '1584.89', 0), ('-FaXLcSFjUI_1560_1620:4', '1584.92', 0), ('-FaXLcSFjUI_1560_1620:4', '1584.95', 0), ('-FaXLcSFjUI_1560_1620:4', '1584.99', 0)]
#         # audio_offset:  1583.89
#         # mid_index:  6
#         # midone:  ('-FaXLcSFjUI_1560_1620:4', '1584.09', 1)
#         # target:  1
#         # target_audio:  1

#         clip_meta_data = cu.generate_clip_meta(entity_metadata, mid_index,
#                                                self.half_clip_length)

#         # clip meta data
#         # [('-FaXLcSFjUI_1560_1620:17', '1596.33', 0),
#         # ('-FaXLcSFjUI_1560_1620:17', '1596.36', 0),
#         # ('-FaXLcSFjUI_1560_1620:17', '1596.4', 0),
#         # ('-FaXLcSFjUI_1560_1620:17', '1596.43', 0),
#         # ('-FaXLcSFjUI_1560_1620:17', '1596.46', 0),
#         # ('-FaXLcSFjUI_1560_1620:17', '1596.5', 0),
#         # ('-FaXLcSFjUI_1560_1620:17', '1596.53', 0),
#         # ('-FaXLcSFjUI_1560_1620:17', '1596.56', 0),
#         # ('-FaXLcSFjUI_1560_1620:17', '1596.6', 0),
#         # ('-FaXLcSFjUI_1560_1620:17', '1596.63', 0),
#         # ('-FaXLcSFjUI_1560_1620:17', '1596.67', 0)]

#         video_data, audio_data = io.load_av_clip_from_metadata(clip_meta_data,
#                                                                self.video_root, self.audio_root, audio_offset,
#                                                                self.target_size)

#         # self.target_size 144

#         if self.do_video_augment:
#             # random flip
#             if bool(random.getrandbits(1)):
#                 video_data = [s.transpose(Image.FLIP_LEFT_RIGHT)
#                               for s in video_data]

#             # random crop
#             width, height = video_data[0].size
#             f = random.uniform(0.5, 1)
#             i, j, h, w = RandomCrop.get_params(
#                 video_data[0], output_size=(int(height*f), int(width*f)))
#             video_data = [s.crop(box=(j, i, w, h)) for s in video_data]

#         if self.video_transform is not None:
#             video_data = [self.video_transform(vd) for vd in video_data]

#         video_data = torch.cat(video_data, dim=0)
#         return (np.float32(audio_data), video_data), target, target_audio


class AudioVideoDatasetAuxLossesForwardPhase(CachedAVSource):
    def __init__(self, target_video, audio_root, video_root, csv_file_path, clip_lenght,
                 target_size, video_transform=None, do_video_augment=False):
        super().__init__()

        # Data directories
        self.audio_root = audio_root
        self.video_root = video_root

        # Post-processing
        self.video_transform = video_transform
        self.do_video_augment = do_video_augment
        self.target_video = target_video

        # Clip arguments
        self.clip_lenght = clip_lenght
        self.half_clip_length = math.floor(self.clip_lenght/2)
        self.target_size = target_size

        self.entity_list = self._cache_entity_data_forward(
            csv_file_path, self.target_video)
        print('len(self.entity_list)', len(self.entity_list))

    def _where_is_ts(self, entity_metadata, ts):
        for idx, val in enumerate(entity_metadata):
            if val[1] == ts:
                return idx

        raise Exception('time stamp not found')

    def __len__(self):
        return int(len(self.entity_list))

    def __getitem__(self, index):
        # Get meta-data
        video_id, entity_id, ts = self.entity_list[index]
        entity_metadata = self.entity_data[video_id][entity_id]

        audio_offset = float(entity_metadata[0][1])
        mid_index = self._where_is_ts(entity_metadata, ts)
        midone = entity_metadata[mid_index]
        gt = midone[-1]

        clip_meta_data = cu.generate_clip_meta(entity_metadata, mid_index,
                                               self.half_clip_length)
        video_data, audio_data = io.load_av_clip_from_metadata(clip_meta_data,
                                                               self.video_root, self.audio_root, audio_offset,
                                                               self.target_size)

        if self.do_video_augment:
            # random flip
            if bool(random.getrandbits(1)):
                video_data = [s.transpose(Image.FLIP_LEFT_RIGHT)
                              for s in video_data]

            # random crop
            width, height = video_data[0].size
            f = random.uniform(0.5, 1)
            i, j, h, w = RandomCrop.get_params(
                video_data[0], output_size=(int(height*f), int(width*f)))
            video_data = [s.crop(box=(j, i, w, h)) for s in video_data]

        if self.video_transform is not None:
            video_data = [self.video_transform(vd) for vd in video_data]

        video_data = torch.cat(video_data, dim=0)
        return np.float32(audio_data), video_data, video_id, ts, entity_id, gt

# ASC Datasets


class ContextualDataset(data.Dataset):
    def get_speaker_context(self, ts_to_entity, video_id, target_entity_id,
                            center_ts, candidate_speakers):
        context_entities = list(ts_to_entity[video_id][center_ts])
        random.shuffle(context_entities)
        context_entities.remove(target_entity_id)

        if not context_entities:  # nos mamamos la lista
            context_entities.insert(0, target_entity_id)  # make sure is at 0
            while len(context_entities) < candidate_speakers:
                context_entities.append(random.choice(context_entities))
        elif len(context_entities) < candidate_speakers:
            context_entities.insert(0, target_entity_id)  # make sure is at 0
            while len(context_entities) < candidate_speakers:
                context_entities.append(random.choice(context_entities[1:]))
        else:
            context_entities.insert(0, target_entity_id)  # make sure is at 0
            context_entities = context_entities[:candidate_speakers]

        return context_entities

    def _decode_feature_data_from_csv(self, feature_data):
        feature_data = feature_data[1:-1]
        feature_data = feature_data.split(',')
        return np.asarray([float(fd) for fd in feature_data])

    def get_time_context(self, entity_data, video_id, target_entity_id,
                         center_ts, half_time_length, stride):
        all_ts = list(entity_data[video_id][target_entity_id].keys())
        center_ts_idx = all_ts.index(str(center_ts))

        start = center_ts_idx-(half_time_length*stride)
        end = center_ts_idx+((half_time_length+1)*stride)
        selected_ts_idx = list(range(start, end, stride))
        selected_ts = []
        for idx in selected_ts_idx:
            if idx < 0:
                idx = 0
            if idx >= len(all_ts):
                idx = len(all_ts)-1
            selected_ts.append(all_ts[idx])

        return selected_ts

    def get_time_indexed_feature(self, video_id, entity_id, selectd_ts):
        time_features = []
        for ts in selectd_ts:
            time_features.append(self.entity_data[video_id][entity_id][ts][0])
        return np.asarray(time_features)

    def _cache_feature_file(self, csv_file):
        entity_data = {}
        feature_list = []
        ts_to_entity = {}

        print('load feature data', csv_file)
        csv_data = io.csv_to_list(csv_file)
        for csv_row in csv_data:
            video_id = csv_row[0]
            ts = csv_row[1]
            entity_id = csv_row[2]
            features = self._decode_feature_data_from_csv(csv_row[-1])
            label = int(float(csv_row[3]))

            # entity_data
            if video_id not in entity_data.keys():
                entity_data[video_id] = {}
            if entity_id not in entity_data[video_id].keys():
                entity_data[video_id][entity_id] = {}
            if ts not in entity_data[video_id][entity_id].keys():
                entity_data[video_id][entity_id][ts] = []
            entity_data[video_id][entity_id][ts] = (features, label)
            feature_list.append((video_id, entity_id, ts))

            # ts_to_entity
            if video_id not in ts_to_entity.keys():
                ts_to_entity[video_id] = {}
            if ts not in ts_to_entity[video_id].keys():
                ts_to_entity[video_id][ts] = []
            ts_to_entity[video_id][ts].append(entity_id)

        print('loaded ', len(feature_list), ' features')
        return entity_data, feature_list, ts_to_entity


class ASCFeaturesDataset(ContextualDataset):
    def __init__(self, csv_file_path, time_lenght, time_stride,
                 candidate_speakers):
        # Space config
        self.time_lenght = time_lenght
        self.time_stride = time_stride
        self.candidate_speakers = candidate_speakers
        self.half_time_length = math.floor(self.time_lenght/2)

        # In memory data
        self.feature_list = []
        self.ts_to_entity = {}
        self.entity_data = {}

        # Load metadata
        self._cache_feature_data(csv_file_path)

    # Parallel load of feature files
    def _cache_feature_data(self, dataset_dir):
        pool = mp.Pool(int(mp.cpu_count()/2))
        files = glob.glob(dataset_dir)
        results = pool.map(self._cache_feature_file, files)
        pool.close()

        for r_set in results:
            e_data, f_list, ts_ent = r_set
            print('unpack ', len(f_list))
            self.entity_data.update(e_data)
            self.feature_list.extend(f_list)
            self.ts_to_entity.update(ts_ent)

    def __len__(self):
        return int(len(self.feature_list))

    def __getitem__(self, index):
        video_id, target_entity_id, center_ts = self.feature_list[index]
        entity_context = self.get_speaker_context(self.ts_to_entity, video_id,
                                                  target_entity_id, center_ts,
                                                  self.candidate_speakers)

        target = self.entity_data[video_id][target_entity_id][center_ts][1]
        feature_set = np.zeros(
            (self.candidate_speakers, self.time_lenght, 1024))
        for idx, ctx_entity in enumerate(entity_context):
            time_context = self.get_time_context(self.entity_data,
                                                 video_id,
                                                 ctx_entity, center_ts,
                                                 self.half_time_length,
                                                 self.time_stride)
            features = self.get_time_indexed_feature(video_id, ctx_entity,
                                                     time_context)
            feature_set[idx, ...] = features

        feature_set = np.asarray(feature_set)
        feature_set = np.swapaxes(feature_set, 0, 2)
        return np.float32(feature_set), target


class ASCFeaturesDatasetForwardPhase(ContextualDataset):
    def __init__(self, csv_file_path, time_lenght, time_stride,
                 candidate_speakers):
        # Space config
        self.time_lenght = time_lenght
        self.time_stride = time_stride
        self.candidate_speakers = candidate_speakers
        self.half_time_length = math.floor(self.time_lenght/2)

        # In memory data
        self.feature_list = []
        self.ts_to_entity = {}
        self.entity_data = {}

        # Single video metdadata
        self.entity_data, self.feature_list, self.ts_to_entity = self._cache_feature_file(
            csv_file_path)

    def __len__(self):
        return int(len(self.feature_list))

    def __getitem__(self, index):
        video_id, target_entity_id, center_ts = self.feature_list[index]
        entity_context = self.get_speaker_context(self.ts_to_entity, video_id,
                                                  target_entity_id, center_ts,
                                                  self.candidate_speakers)

        feature_set = np.zeros(
            (self.candidate_speakers, self.time_lenght, 1024))
        for idx, ctx_entity in enumerate(entity_context):
            time_context = self.get_time_context(self.entity_data,
                                                 video_id,
                                                 ctx_entity, center_ts,
                                                 self.half_time_length,
                                                 self.time_stride)
            features = self.get_time_indexed_feature(video_id, ctx_entity,
                                                     time_context)
            feature_set[idx, ...] = features

        feature_set = np.asarray(feature_set)
        feature_set = np.swapaxes(feature_set, 0, 2)
        return np.float32(feature_set), video_id, center_ts, target_entity_id
