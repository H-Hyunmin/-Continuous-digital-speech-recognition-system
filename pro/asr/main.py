# 导入音频特征提取工具包
import torch
import sys
from torch.utils.data import DataLoader
from utils import loader ,load_audio,VAD,load_wav
from Model import Audio, train_model, infer_model, infer_wav


model = Audio()
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')



if __name__ == "__main__":
    # 根据命令行参数决定
    if len(sys.argv) > 1:
        if sys.argv[1] == 'infer':
            test_data=loader('data/test.tsv')
            test_loader=DataLoader(test_data,batch_size=100,shuffle=False)
            # 加载模型
            model.load_state_dict(torch.load('./model/best_model.ckpt'))
            # 进行推理
            infer_model(model, test_loader)
        elif sys.argv[1] == 'audio':
            # 加载模型
            model.load_state_dict(torch.load('./model/best_model.ckpt'))
            # 读取音频文件
            audio_datas = []
            audio_datas = load_audio(audio_datas,'fd395b74_nohash_0.wav',7)
            audio_datas = load_audio(audio_datas,'fd395b74_nohash_1.wav',7)
            audio_datas = load_audio(audio_datas,'fd395b74_nohash_2.wav',7)
            audio_datas = load_audio(audio_datas,'fd395b74_nohash_3.wav',7)
            audio_datas = load_audio(audio_datas,'fd395b74_nohash_4.wav',7)

            audio_loader=DataLoader(audio_datas,batch_size=1,shuffle=False)
            # 进行推理
            infer_model(model, audio_loader)
        elif sys.argv[1] == 'train':
            train_data=loader('data/train.tsv')
            dev_data=loader('data/dev.tsv')
            train_loader=DataLoader(train_data,batch_size=64,shuffle=True)
            dev_loader=DataLoader(dev_data,batch_size=64,shuffle=False)
            train_model(model, train_loader, dev_loader,13)

        elif sys.argv[1] == 'recognize':
            wavname=sys.argv[2]
            audios, fs = VAD(wavname, 3)
            datas = load_wav(audios,fs)
            datas = DataLoader(datas, batch_size=100, shuffle=False)
            model.load_state_dict(torch.load('./model/best_model.ckpt'))
            infer_wav(model, datas)
        elif sys.argv[1] == 'gui':
            import gui

