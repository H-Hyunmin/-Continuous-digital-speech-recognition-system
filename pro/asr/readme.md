# 语音识别项目

这是一个基于深度学习的语音识别项目，使用Python语言编写。项目主要包括音频特征提取、模型训练、模型推理和GUI界面等部分。

## 项目结构

### 数据集较大，只附上了部分数据

- `data`：下存放数据集和标签。分别存放在对应的文件夹中。数据来源开源数据集
- `model`：存放已训练好的模型。
- `utils.py`：包含音频特征提取和数据加载等工具函数。
- `gui.py`：包含一个基于tkinter的GUI界面，用户可以通过这个界面进行音频文件的选择、播放、录音、音频特征的可视化和语音识别等操作。
- `Model.py`：定义了语音识别模型。
- `main.py`：项目的主程序，根据命令行参数进行不同的操作，如模型训练、模型推理、音频文件的语音识别等。

## 项目环境

项目torch版本：2.3.0+cpu
安装依赖：`pip install -r requirements.txt`

## 使用方法

1. 数据集处理：运行utils.py下的`split_data2`函数，将./data下的数据集划分为训练集、验证集和测试集。测试集的数量可在函数中修改。
2. 训练模型：`python main.py train` 会覆盖已有模型，请做好备份
3. 测试集推理：`python main.py infer`
4. 音频文件的语音识别：`python main.py recognize your_wav_file_name.wav`;
5. 启动GUI界面：`python main.py gui`;
