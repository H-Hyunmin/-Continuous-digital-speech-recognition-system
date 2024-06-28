# utils.py

import numpy as np
import scipy.io.wavfile as wav
from python_speech_features import mfcc
import os
import random
from torch.utils.data import Dataset
import webrtcvad

# MFCC特征提取
def get_mfcc(data, fs):
    # # MFCC特征提取
    # wav_feature =  mfcc(data, fs)
    # # 特征一阶差分
    # d_mfcc_feat = delta(wav_feature, 1)
    # # 特征二阶差分
    # d_mfcc_feat2 = delta(wav_feature, 2)
    # # 特征拼接
    # feature = np.concatenate([wav_feature.reshape(1, -1, 13), d_mfcc_feat.reshape(1, -1, 13), d_mfcc_feat2.reshape(1, -1, 13)], 0)
    # MFCC特征提取
    wav_feature =  mfcc(data, fs,numcep=6)
    # 特征重塑
    feature = wav_feature.reshape(1, -1, 6)

    # 对数据进行截取或者填充
    if feature.shape[1]>64:
        feature = feature[:, :64, :]
    else:
        feature = np.pad(feature, ((0, 0), (0, 64-feature.shape[1]), (0, 0)), 'constant')
    # 通道转置(HWC->CHW)
    feature = feature.transpose((2, 0, 1))
    # 新建空维度(CHW->NCHW)
    return feature

class AudioDataset(Dataset):
    def __init__(self, tsv_file):
        self.data = []
        with open(tsv_file, 'r', encoding='UTF-8') as f:
            for line in f:
                audio, label = line[:-1].split('\t')
                self.data.append((audio, int(label)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio, label = self.data[idx]
        fs, signal = wav.read('work/' + audio)  # 读取音频文件
        feature = get_mfcc(signal, fs)
        return feature, label



def load_audio(datas,audio_file,label):
    fs, signal = wav.read(audio_file)  # 读取音频文件
    feature = get_mfcc(signal, fs)  # 获取MFCC特征
    datas.append([feature, label])
    return datas

def load_wav(audios,fs):
    datas=[]
    for audio in audios:
        signal = audio
        feature = get_mfcc(signal, fs)
        datas.append([feature, 0])
    return datas


# 读取数据列表文件
def loader(tsv):
    datas = []
    with open(tsv, 'r', encoding='UTF-8') as f:
        for line in f:
            audio, label = line[:-1].split('\t')            
            fs, signal = wav.read('data/'+audio) # 读取音频文件
            feature = get_mfcc(signal, fs)
            datas.append([feature, int(label)])
    return datas

# 数据读取器
def reader(datas, batch_size, is_random=True):
    features = []
    labels = []
    if is_random:
        random.shuffle(datas)
    for data in datas:
        feature, label = data
        features.append(feature)
        labels.append(label)
        if len(labels)==batch_size:
            features = np.concatenate(features, 0).reshape(-1, 13, 3, 64).astype('float32')
            labels = np.array(labels).reshape(-1, 1).astype('int64')
            yield features, labels
            features = []
            labels = []

# 数据划分函数
def split_data1():
    # 读取所有音频数据
    recordings = ['recordings/'+_ for _ in os.listdir('work/recordings')]
    total = []
    for recording in recordings:
        label = int(recording[11])
        total.append('%s\t%s\n' % (recording, label))

    train = open('work/train.tsv', 'w', encoding='UTF-8')
    dev = open('work/dev.tsv', 'w', encoding='UTF-8')
    test = open('work/test.tsv', 'w', encoding='UTF-8')

    random.shuffle(total)
    split_num = int((len(total)-100)*0.9)
    for line in total[:split_num]:
        train.write(line)
    for line in total[split_num:-100]:
        dev.write(line)
    for line in total[-100:]:
        test.write(line)

    train.close()
    dev.close()
    test.close()

# 数据划分函数
def split_data2():
    # 读取所有音频数据
    total = []
    for i in range(10):  # 对于每个数字
        dir_name = str(i)  # 文件夹名称
        recordings = [dir_name + '/' + _ for _ in os.listdir('data/' + dir_name)]
        for recording in recordings:
            total.append('%s\t%s\n' % (recording, i))  # 标签就是文件夹的名字

    train = open('data/train.tsv', 'w', encoding='UTF-8')
    dev = open('data/dev.tsv', 'w', encoding='UTF-8')
    test = open('data/test.tsv', 'w', encoding='UTF-8')

    random.shuffle(total)
    split_num = int((len(total)-10)*0.9)
    for line in total[:split_num]:
        train.write(line)
    for line in total[split_num:-10]:
        dev.write(line)
    for line in total[-10:]:
        test.write(line)

    train.close()
    dev.close()
    test.close()

def split_data():
    # 读取部分
    total = []
    for i in range(10):  # 对于每个数字
        dir_name = str(i)  # 文件夹名称
        recordings = [dir_name + '/' + _ for _ in os.listdir('data/' + dir_name)[:2000]]  # 只读取前2000个文件
        for recording in recordings:
            total.append('%s\t%s\n' % (recording, i))  # 标签就是文件夹的名字

    train = open('data/train.tsv', 'w', encoding='UTF-8')
    dev = open('data/dev.tsv', 'w', encoding='UTF-8')
    test = open('data/test.tsv', 'w', encoding='UTF-8')

    random.shuffle(total)
    split_num = int((len(total)-1000)*0.9)
    for line in total[:split_num]:
        train.write(line)
    for line in total[split_num:-1000]:
        dev.write(line)
    for line in total[-1000:]:
        test.write(line)

    train.close()
    dev.close()
    test.close()

def VAD(audio, mode):
    # 读取音频
    fs, signal = wav.read(audio)
    # vad初始化
    vad = webrtcvad.Vad()
    vad.set_mode(mode)
    # 数据填充
    padding = int(fs*0.02) - (signal.shape[0] % int(fs*0.02))
    if padding < 0:
        padding += int(fs*0.02)
    signal = np.pad(signal, (0, padding), 'constant')
    # 数据分帧
    lens = signal.shape[0]
    signals = np.split(signal, lens//int(fs*0.02))
    # 音频切分
    audio = []
    audios = []
    for signal in signals:
        if vad.is_speech(signal, fs):
            audio.append(signal)
        elif len(audio) and (not vad.is_speech(signal, fs)):
            audios.append(np.concatenate(audio, 0))
            audio = []
    return audios, fs


if __name__ == "__main__":
    split_data2()