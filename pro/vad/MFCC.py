import numpy as np
import librosa
from sklearn.cluster import KMeans
from evaluate import evaluate
import time
import noisereduce as nr
from sklearn.mixture import GaussianMixture

# 定义参数
FRAME_SIZE = 2048
HOP_SIZE = 512
NUM_MFCC = 4  # MFCC系数数量512
NUM_CLUSTERS = 2  # KMeans聚类数量

SAMPLE_RATE = 8000
WAV = './test_data/data/data_3_noise.wav'
LABEL_INPUT = './test_data/label/data_3.txt'
PREDICT_INPUT = './test_data/predict/data_3_noise.txt'

# 加载音频文件
def load_audio(file_path, sr=8000):
    # 使用librosa读取音频
    audio, sr = librosa.load(file_path, sr=sr)
    # 应用噪声抑制
    audio = nr.reduce_noise(y=audio, sr=sr)

    return audio, sr

# # 提取MFCC特征
def extract_mfcc(audio, sr):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=NUM_MFCC)
    return mfccs.T  # 转置矩阵，使得每行对应一个时间段

# 进行KMeans聚类
def perform_clustering(features):
    kmeans = KMeans(n_clusters=NUM_CLUSTERS)
    labels = kmeans.fit_predict(features)
    return labels

# 进行GMM聚类
# def perform_clustering(features):
#     gmm = GaussianMixture(n_components=NUM_CLUSTERS)
#     labels = gmm.fit_predict(features)
#     return labels

# 获取语音段落
def get_voiced_segments(audio, labels):
    voiced_segments = []

    # 计算每个聚类的平均能量
    energy_0 = np.mean([np.sum(audio[i*HOP_SIZE:(i+1)*HOP_SIZE]**2) for i in range(len(labels)) if labels[i] == 0])
    energy_1 = np.mean([np.sum(audio[i*HOP_SIZE:(i+1)*HOP_SIZE]**2) for i in range(len(labels)) if labels[i] == 1])

    # 确定哪个聚类代表“无声”
    silence_label = 0 if energy_0 < energy_1 else 1

    start_idx = None
    for i in range(1, len(labels)):
        if labels[i] != labels[i-1]:  # 如果当前帧与前一帧标签不同
            if labels[i-1] == silence_label:  # 如果前一帧标签为无声标签，表示是无声段，当前帧是语音开始
                start_idx = i * HOP_SIZE
            elif start_idx is not None:  # 如果前一帧标签为语音标签，表示是语音段，当前帧是语音结束
                end_idx = i * HOP_SIZE
                voiced_segments.append((start_idx, end_idx))  # 保存起始和结束索引
                start_idx = None  # 重置start_idx
    # 处理最后一段语音
    if start_idx is not None:
        end_idx = len(audio)
        voiced_segments.append((start_idx, end_idx))  # 保存起始和结束索引
    return voiced_segments

# 将语音段落保存到txt文件中
def save_segments_to_txt(voiced_segments, output_path):
    with open(output_path, 'w') as file:
        for start_idx, end_idx in voiced_segments:
            file.write(f"{start_idx},{end_idx}\n")  # 按照指定的格式打印索引

# 测试
def eval():
    wav_input,sample_rate = librosa.load(WAV,sr=SAMPLE_RATE)
    data_length = len(wav_input)
    label_input = LABEL_INPUT
    predict_input = PREDICT_INPUT

    f1_score,accuracy,recall,precision = evaluate(data_length, label_input, predict_input)
    print('test:'+WAV)
    print('f1_score: ',f1_score)
    print('accuracy: ',accuracy)
    print('recall: ',recall)
    print('precision: ',precision)
    print('\n')


def is_silence(audio, sr, energy_threshold=100, zcr_threshold=0.05):
    # 计算音频的能量
    energy = np.sum(audio ** 2)
    
    # 计算零交叉率
    zcr = librosa.feature.zero_crossing_rate(audio)[0, 0]
    
    # 如果能量和零交叉率都低于阈值，则认为音频是噪声
    return energy < energy_threshold and zcr < zcr_threshold


def merge_close_segments(voiced_segments, threshold=1024):
    merged_segments = []
    start_idx, end_idx = voiced_segments[0]
    for next_start_idx, next_end_idx in voiced_segments[1:]:
        # 如果下一个语音段的开始索引与当前语音段的结束索引之间的差值小于阈值
        if next_start_idx - end_idx <= threshold:
            # 将下一个语音段的结束索引设置为当前语音段的结束索引
            end_idx = next_end_idx
        else:
            # 否则，保存当前语音段，并开始一个新的语音段
            merged_segments.append((start_idx, end_idx))
            start_idx, end_idx = next_start_idx, next_end_idx
    # 保存最后一个语音段
    merged_segments.append((start_idx, end_idx))
    return merged_segments




# 主函数
def main(input_path, output_path):
    # 加载音频文件
    audio, sr = load_audio(input_path)


    if is_silence(audio, sr):
        print("May no voice detected in the audio.")
        return

    # 提取MFCC特征
    mfcc_features = extract_mfcc(audio, sr)

    # 进行KMeans聚类
    labels = perform_clustering(mfcc_features)
    np.savetxt('py_data1_lables.txt', labels, fmt='%d', delimiter=' ')
    # 获取语音段落
    voiced_segments = get_voiced_segments(audio, labels)

    # 使用后处理函数
    voiced_segments = merge_close_segments(voiced_segments, threshold=512)

    # 将语音段落保存到txt文件中
    save_segments_to_txt(voiced_segments, output_path)

    # # 评估
    eval()

# 测试
if __name__ == "__main__":
    start_time = time.time()  # 记录开始时间

    input_path = WAV
    output_path = PREDICT_INPUT 
    main(input_path, output_path)

    end_time = time.time()  # 记录结束时间
    print("程序运行时间：{}秒".format(end_time - start_time))


