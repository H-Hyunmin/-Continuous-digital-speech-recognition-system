import tkinter as tk
from tkinter import filedialog, ttk
import scipy.io.wavfile as wav
from utils import VAD,load_wav
import matplotlib.pyplot as plt
import torch
from Model import  infer_wav,Audio
from torch.utils.data import DataLoader
import sounddevice as sd
from scipy.io.wavfile import write,read

import time


plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.master.geometry("600x200")  # 设置窗口的初始大小
        self.pack()
        self.create_widgets()
        self.record_thread = None  # Initialize record_thread
        self.recording=False
        self.myrecording = None
        self.fs = 16000  # Sample rate

    def create_widgets(self):
        self.master.title("语音识别小程序")
        
        button_frame = ttk.Frame(self)
        button_frame.pack(side="left")

        self.select_button = ttk.Button(button_frame)
        self.select_button["text"] = "选择音频文件"
        self.select_button["command"] = self.select_file
        self.select_button.pack(side="top")

        self.play_button = ttk.Button(button_frame)
        self.play_button["text"] = "播放音频"
        self.play_button["command"] = self.play_audio
        self.play_button.pack(side="top")

        self.record_button = ttk.Button(button_frame)
        self.record_button["text"] = "开始录音"
        self.record_button["command"] = self.start_recording
        self.record_button.pack(side="top")

        self.stop_button = ttk.Button(button_frame)
        self.stop_button["text"] = "停止录音"
        self.stop_button["command"] = self.stop_recording
        self.stop_button.pack(side="top")

        self.plot_button = ttk.Button(button_frame)
        self.plot_button["text"] = "绘制时域图与频域图"
        self.plot_button["command"] = self.plot_waveform
        self.plot_button.pack(side="top")


        self.plot_vad_button = ttk.Button(button_frame)
        self.plot_vad_button["text"] = "绘制VAD分段时域图"
        self.plot_vad_button["command"] = self.plot_vad_waveform
        self.plot_vad_button.pack(side="top")

        self.recognize_button = ttk.Button(button_frame)
        self.recognize_button["text"] = "开始识别"
        self.recognize_button["command"] = self.recognize_speech
        self.recognize_button.pack(side="top")

        self.result_text = tk.Text(self, height=14, width=50)
        self.result_text.pack(side="right")



    def select_file(self):
        self.filename = filedialog.askopenfilename(filetypes=[("音频文件", "*.wav")])
        self.result_text.insert(tk.END, "选择的文件: " + self.filename + "\n")

    def play_audio(self):
        if hasattr(self, 'filename'):
            fs, data = read(self.filename)
            sd.play(data, fs)  # Play the audio
            self.result_text.insert(tk.END, "开始播放音频...\n")
        else:
            self.result_text.insert(tk.END, "请先选择音频文件\n")

    def plot_waveform(self):
        if hasattr(self, 'filename'):
            samplerate, data = wav.read(self.filename)
            plt.figure(figsize=(10, 4))
            plt.plot(data)
            plt.title("时域图")
            plt.show()
        else:
            self.result_text.insert(tk.END, "请先选择音频文件\n")

    def plot_vad_waveform(self):
        if hasattr(self, 'filename'):
            aduios,fs=VAD(self.filename,3)
            fig, axs = plt.subplots(nrows=1, ncols=len(aduios), figsize=(10, 4))
            for i, segment in enumerate(aduios):
                axs[i].plot(segment)
                axs[i].set_title(f"VAD Segment {i+1}")
            plt.show()
        else:
            self.result_text.insert(tk.END, "请先选择音频文件\n")

    def start_recording(self):
        self.recording = True
        self.start_time = time.time()  # Record the start time
        self.myrecording = sd.rec(int(self.fs * 10), samplerate=self.fs, channels=2)
        self.result_text.insert(tk.END, "开始录音...\n")

    def stop_recording(self):
        if self.recording:
            sd.stop()
            record_time = time.time() - self.start_time  # Calculate the actual record time
            self.myrecording = self.myrecording[:int(self.fs * record_time)]  # Trim the recording to the actual length
            write('output.wav', self.fs, self.myrecording)  # Save as WAV file 
            self.result_text.insert(tk.END, "录音结束，已保存为output.wav\n")
            self.recording = False
        else:
            self.result_text.insert(tk.END, "没有正在进行的录音。\n")



    def recognize_speech(self):
        model = Audio()
        model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
        wavname=self.filename
        audios, fs = VAD(wavname, 3)
        datas = load_wav(audios,fs)
        datas = DataLoader(datas, batch_size=100, shuffle=False)
        model.load_state_dict(torch.load('./model/best_model.ckpt'))
        pre=infer_wav(model, datas)
        self.result_text.insert(tk.END, "识别结果为："+str(pre)+"\n")


root = tk.Tk()
app = Application(master=root)
app.mainloop()
