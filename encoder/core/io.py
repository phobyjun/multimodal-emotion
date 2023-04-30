import os
from PIL import Image
from scipy.io import wavfile
from core.util import Logger
import numpy as np
import python_speech_features
import csv
import time
import json


def custom_ts_to_seconds(td):
    ts = td.split('-')[-3:]

    seconds = 0
    seconds = float(ts[0])//100*3600 + float(ts[0]) % 100 * \
        60 + float(ts[1]) + float(ts[2])*0.001

    return seconds


def preprocessRGBData(rgb_data):
    rgb_data = rgb_data.astype('float32')
    rgb_data = rgb_data/255.0
    rgb_data = rgb_data - np.asarray((0.485, 0.456, 0.406))

    return rgb_data


def _pil_loader(path, target_size):
    try:
        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.resize(target_size)
            return img.convert('RGB')
    except OSError as e:
        return Image.new('RGB', target_size)


def set_up_log_and_ws_out(models_out, opt_config, experiment_name, headers=None):
    target_logs = os.path.join(models_out, experiment_name + '_logs.csv')
    target_models = os.path.join(models_out, experiment_name)
    print('target_models', target_models)
    if not os.path.isdir(target_models):
        os.makedirs(target_models)
    log = Logger(target_logs, ';')

    if headers is None:
        log.writeHeaders(['epoch', 'train_loss', 'train_auc', 'train_map',
                          'val_loss', 'val_auc', 'val_map'])
    else:
        log.writeHeaders(headers)

    # Dump cfg to json
    dump_cfg = opt_config.copy()
    for key, value in dump_cfg.items():
        if callable(value):
            try:
                dump_cfg[key] = value.__name__
            except:
                dump_cfg[key] = 'CrossEntropyLoss'
    json_cfg = os.path.join(models_out, experiment_name+'_cfg.json')
    with open(json_cfg, 'w') as json_file:
        json.dump(dump_cfg, json_file)

    models_out = os.path.join(models_out, experiment_name)
    return log, models_out


def csv_to_list(csv_path):
    as_list = None
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        as_list = list(reader)
    return as_list


def _generate_mel_spectrogram(audio_clip, sample_rate):
    mfcc = zip(*python_speech_features.mfcc(audio_clip, sample_rate))
    audio_features = np.stack([np.array(i) for i in mfcc])
    audio_features = np.expand_dims(audio_features, axis=0)
    return audio_features


def _fit_audio_clip(audio_clip, sample_rate, video_clip_lenght):
    target_audio_length = (1.0/4.0)*sample_rate*video_clip_lenght
    # print("video_clip_lenght: ", video_clip_lenght)
    # print("target_audio_length: ", target_audio_length)

    pad_required = int((target_audio_length-len(audio_clip))/2)
    # print("pad_required: ", pad_required)

    # print("before len(audio_clip): ", len(audio_clip))

    if pad_required > 0:
        audio_clip = np.pad(audio_clip, pad_width=(pad_required, pad_required),
                            mode='reflect')
    if pad_required < 0:
        audio_clip = audio_clip[-1*pad_required:pad_required]

    # print("after len(audio_clip): ", len(audio_clip))

    return audio_clip


def load_av_clip_from_metadata(clip_meta_data, frames_source, audio_source,
                               audio_offset, target_size):

    # clip_meta_data, frames_source, audio_source, audio_offset, target_size
    # clip_meta_data: list of tuples (entity_id, timestamp)
    # frames_source: VIDEO PATH
    # audio_source: AUDIO PATH
    # audio_offset: video start time
    # target_size: (144, 144)

    # time stamp sequence in clip
    ts_sequence = [str(meta[1]) for meta in clip_meta_data]

    min_ts = float(clip_meta_data[0][1])
    max_ts = float(clip_meta_data[-1][1])
    entity_id = clip_meta_data[0][0]

    # print("frames_source, entity_id, ts+'.jpg': ",
    #       frames_source, entity_id, "ts"+'.jpg')

    selected_frames = [os.path.join(
        frames_source, entity_id, ts+'.jpg') for ts in ts_sequence]
    video_data = [_pil_loader(sf, target_size) for sf in selected_frames]
    audio_file = os.path.join(audio_source, entity_id+'.wav')

    try:
        sample_rate, audio_data = wavfile.read(audio_file)
    except:
        sample_rate, audio_data = 16000,  np.zeros((16000*10))

    audio_start = int((min_ts-audio_offset)*sample_rate)
    audio_end = int((max_ts-audio_offset)*sample_rate)
    audio_clip = audio_data[audio_start:audio_end]

    # print("============================")
    # print("ts_sequence: ", ts_sequence)
    # print("max_ts - min_ts: ", max_ts - min_ts)
    # print("audio_start: ", audio_start)
    # print("audio_end: ", audio_end)
    # print("end-start: ", audio_end-audio_start)
    # print("sample_rate: ", sample_rate)
    # print("len(audio_data): ", len(audio_data))
    # print("len(audio_clip): ", len(audio_clip))
    # print("============================")

    # print("min_ts: ", min_ts)
    # print("max_ts: ", max_ts)
    # print("audio_offset: ", audio_offset)
    # print("audio_start: ", audio_start)
    # print("audio_end: ", audio_end)
    # print("len(audio_clip): ", len(audio_clip))

    # min_ts:  1142.83
    # max_ts:  1143.16
    # audio_offset:  1142.06
    # audio_start:  12319
    # audio_end:  17600
    # len(audio_clip):  5281
    # min_ts:  1317.89
    # max_ts:  1318.22

    # print("len(audio_data): ", len(audio_data))
    # print("sample_rate*(2/25): ", sample_rate*(2/25))
    # print("len(audio_clip): ", len(audio_clip))
    # print("audio_start, audio_end: ", audio_start, audio_end)

    if len(audio_clip) < sample_rate*(2/25):
        audio_clip = np.zeros((int(sample_rate*(len(clip_meta_data)/25))))

    # print("(int(sample_rate*(len(clip_meta_data)/25))): ", (int(sample_rate*(len(clip_meta_data)/25))))

    # print("len(clip_meta_data): ", len(clip_meta_data))

    # len(audio_data):  160000
    # sample_rate*(2/25):  1280.0
    # len(audio_clip):  5441
    # audio_start, audio_end:  4799 10240
    # (int(sample_rate*(len(clip_meta_data)/25))):  7040
    # len(audio_data):  160000
    # sample_rate*(2/25):  1280.0
    # len(audio_clip):  5280
    # audio_start, audio_end:  18079 23359
    # (int(sample_rate*(len(clip_meta_data)/25))):  7040

    # print("sample_rate: ", sample_rate) # 16000

    # print("len(selected_frames): ", len(selected_frames))

    audio_clip = _fit_audio_clip(audio_clip, sample_rate, len(selected_frames))
    # print("after fit len(audio_clip): ", len(audio_clip))

    audio_features = _generate_mel_spectrogram(audio_clip, sample_rate)

    return video_data, audio_features


def load_av_clip_from_metadata_TEST(clip_meta_data, audio_source,
                                    audio_offset, target_seg, clip_length):

    # print("clip_meta_data: ", clip_meta_data)

    # clip_meta_data, frames_source, audio_source, audio_offset, target_size
    # clip_meta_data: list of tuples (entity_id, timestamp)
    # frames_source: VIDEO PATH
    # audio_source: AUDIO PATH
    # audio_offset: video start time
    # target_size: (144, 144)

    audio_offset = custom_ts_to_seconds(audio_offset)

    # temp = [meta['ts'] for meta in clip_meta_data]
    # print("temp: ", temp)

    # time stamp sequence in clip
    ts_sequence = [custom_ts_to_seconds(meta['ts']) for meta in clip_meta_data]

    min_ts = float(ts_sequence[0])
    max_ts = float(ts_sequence[-1])

    # print("ts_sequence: ", ts_sequence)
    # print("min_ts: ", min_ts)
    # print("max_ts: ", max_ts)

    target_seg_folder = target_seg.split('_')[0].replace('Sess', 'Session')

    # print("audio_source: ", audio_source)

    audio_file = os.path.join(
        audio_source, target_seg_folder, target_seg + '.wav')

    # print("audio_file: ", audio_file)

    try:
        sample_rate, audio_data = wavfile.read(audio_file)
        # print("::try::")
    except Exception as e:
        # print("except error: ", e)
        sample_rate, audio_data = 16000,  np.zeros((16000*10))
        # print("::except::")

    audio_start = int((min_ts-audio_offset))
    audio_end = int((max_ts-audio_offset))

    # print("=========================== io ===========================")
    # print("int((min_ts-audio_offset)): ", int((min_ts-audio_offset)))
    # print("int((max_ts-audio_offset)): ", int((max_ts-audio_offset)))
    # print("int(audio_start), int(audio_end): ",
    #       int(audio_start), int(audio_end))
    # print("len(audio_data): ", len(audio_data))
    # print("sample_rate: ", sample_rate)
    # print("audio_data: ", audio_data)

    audio_start = int((min_ts-audio_offset)*sample_rate)
    audio_end = int((max_ts-audio_offset)*sample_rate)
    audio_clip = audio_data[audio_start:audio_end]

    # print("============================")
    # print("ts_sequence: ", ts_sequence)
    # print("max_ts - min_ts: ", max_ts - min_ts)
    # print("audio_start: ", audio_start)
    # print("audio_end: ", audio_end)
    # print("end-start: ", audio_end-audio_start)
    # print("sample_rate: ", sample_rate)
    # print("len(audio_data): ", len(audio_data))
    # print("len(audio_clip): ", len(audio_clip))
    # print("============================")

    # print("min_ts: ", min_ts)
    # print("max_ts: ", max_ts)
    # print("audio_offset: ", audio_offset)
    # print("audio_start: ", audio_start)
    # print("audio_end: ", audio_end)
    # print("len(audio_clip): ", len(audio_clip))

    # min_ts:  1142.83
    # max_ts:  1143.16
    # audio_offset:  1142.06
    # audio_start:  12319
    # audio_end:  17600
    # len(audio_clip):  5281
    # min_ts:  1317.89
    # max_ts:  1318.22

    # print("len(audio_data): ", len(audio_data))
    # print("sample_rate*(2/25): ", sample_rate*(2/25))
    # print("len(audio_clip): ", len(audio_clip))
    # print("audio_start, audio_end: ", audio_start, audio_end)

    if len(audio_clip) < sample_rate*(2/25):
        audio_clip = np.zeros((int(sample_rate*(len(clip_meta_data)/25))))

    # print("(int(sample_rate*(len(clip_meta_data)/25))): ", (int(sample_rate*(len(clip_meta_data)/25))))

    # print("len(clip_meta_data): ", len(clip_meta_data))

    # len(audio_data):  160000
    # sample_rate*(2/25):  1280.0
    # len(audio_clip):  5441
    # audio_start, audio_end:  4799 10240
    # (int(sample_rate*(len(clip_meta_data)/25))):  7040
    # len(audio_data):  160000
    # sample_rate*(2/25):  1280.0
    # len(audio_clip):  5280
    # audio_start, audio_end:  18079 23359
    # (int(sample_rate*(len(clip_meta_data)/25))):  7040

    # print("sample_rate: ", sample_rate) # 16000

    # print("len(selected_frames): ", len(selected_frames))

    audio_clip = _fit_audio_clip(audio_clip, sample_rate, clip_length)
    # print("after fit len(audio_clip): ", len(audio_clip))
    audio_features = _generate_mel_spectrogram(audio_clip, sample_rate)

    return audio_features
