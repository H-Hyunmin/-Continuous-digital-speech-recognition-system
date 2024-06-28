import numpy as np
import librosa

SAMPLE_RATE = 8000
WAV = 'test_data/data/data_1.wav'
LABEL_INPUT = 'test_data/label/data_1.txt'
PREDICT_INPUT = 'test_data/predict/data1_result.txt'

def evaluate(data_length, label_input, predict_input):

    voice_length = 0
    predict_voice_length = 0

    label = np.full(data_length, 0)
    label_data = np.loadtxt(label_input,delimiter=',') 
    for i in range(len(label_data)):
        a = int(label_data[i][0])
        b = int(label_data[i][1])
        label[a:b+1] = 1
        voice_length += (b - a)

    predict = np.full(data_length, 0)
    predict_data = np.loadtxt(predict_input,delimiter=',') 
    for i in range(len(predict_data)):
        a = int(predict_data[i][0])
        b = int(predict_data[i][1])
        predict[a:b+1] = 1
        predict_voice_length += (b - a)

    false_detection = 0
    miss_detection = 0
    acc = 0
    tp = 0
    for i in range(data_length):
        if label[i] == 0 and predict[i] == 1:
            false_detection += 1
        if label[i] == 1 and predict[i] == 0:
            miss_detection += 1
        if label[i] == predict[i]:
            acc += 1
        if label[i] == 1 and predict[i] == 1:
            tp += 1
    
    accuracy = acc/data_length
    recall = tp/voice_length
    precision = tp/predict_voice_length
    f1_score = (2*precision*recall)/(precision+recall)

    return f1_score,accuracy,recall,precision
    
if __name__ == '__main__':  
    
    wav_input,sample_rate = librosa.load(WAV,sr=SAMPLE_RATE)
    data_length = len(wav_input)
    label_input = LABEL_INPUT
    predict_input = PREDICT_INPUT

    f1_score,accuracy,recall,precision = evaluate(data_length, label_input, predict_input)
    print('\n')
    print('f1_score: ',f1_score)
    print('accuracy: ',accuracy)
    print('recall: ',recall)
    print('precision: ',precision)
    print('\n')