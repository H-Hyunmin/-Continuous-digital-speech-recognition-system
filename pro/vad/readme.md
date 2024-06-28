# MFCC特征提取和KMeans聚类

这个项目是一个音频处理程序，它使用Mel频率倒谱系数（MFCC）特征和KMeans聚类来检测音频中的语音段落。

# 测试说明

该算法对于正负样本接近1:1的比例分布的有人声音频效果较好，且有较好的抗噪性和鲁棒性。

由于Kmeans聚类的限制，该算法对于纯噪声无人声或者人声占比极低的数据测试效果可能不佳，由于时间原因尚未进行更进一步优化。

./data 文件夹下存放了我们的一些测试样本。

# 代码结构

主要的Python文件是MFCC.py，它包含以下函数：

load_audio(file_path, sr=8000): 加载音频文件并应用噪声抑制。

extract_mfcc(audio, sr): 提取音频的MFCC特征。

perform_clustering(features): 对MFCC特征进行KMeans聚类。

get_voiced_segments(audio, labels): 根据聚类结果获取语音段落。

save_segments_to_txt(voiced_segments, output_path): 将语音段落保存到txt文件中。

is_silence(audio, sr, energy_threshold=100, zcr_threshold=0.05): 判断音频是否为纯噪声无人声。

merge_close_segments(voiced_segments, threshold=1024): 合并接近的语音段落。

main(input_path, output_path): 主函数，调用上述函数进行音频处理。

# 使用方法

安装所需的Python库：numpy, librosa, sklearn, noisereduce。

修改MFCC.py中的WAV、LABEL_INPUT和PREDICT_INPUT变量，使它们指向你的音频文件和标签文件。

运行MFCC.py：python MFCC.py。
# 输出

程序将在指定的输出路径生成一个txt文件，其中包含检测到的语音段落的起始和结束索引。并在终端直接打印出测试结果如：
```
test:./test_data/data/data_1.wav
f1_score:  0.9477916136012488
accuracy:  0.9738944796603357
recall:  0.9155771529947163
precision:  0.9823556673728814
```

