import cv2
import numpy as np
import matplotlib.pyplot as plt
import librosa
from scipy.signal import correlate


def calculate_rms(image):
    # 将图像转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 计算均方根值
    rms = np.sqrt(np.mean(gray.astype(float) ** 2))
    return rms

# 视频文件路径
video_path = '/home/ma-user/work/project/Hear-Your-Click/result/tmp/dog_cat.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# 存储RMS值
rms_values_v = []
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break  # 如果没有更多帧，则退出循环
    
    # 计算当前帧的RMS值
    rms = calculate_rms(frame)
    rms_values_v.append(rms)
    
    # 打印进度
    frame_count += 1
    print(f"Processed {frame_count} frames")

cap.release()

avg = np.mean(np.array(rms_values_v))

# 绘制RMS曲线
plt.figure(figsize=(10, 5))
plt.plot(rms_values_v, label='RMS Value')
plt.title(f'RMS Curve of Video Over Time,avg={avg}')
plt.xlabel('Frame Number')
plt.ylabel('RMS Value')
plt.legend()
plt.savefig('./rms_video.png')


for i in range(10):
    # 加载音频文件
    audio_path = f'/home/ma-user/work/project/Hear-Your-Click/result/df_output/sample_{i}_diff.wav'
    y, sr = librosa.load(audio_path, sr=None)  # sr=None表示使用原始采样率

    # 计算帧长和帧移
    frame_length = 2048  # 每个分析窗口的长度
    hop_length = 512    # 相邻窗口之间的跳跃步长

    # 计算RMS能量
    rms_values_a = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)

    # 将RMS值转换为一维数组
    rms_values_a = rms_values_a[0]

    # 创建时间轴
    frames = range(len(rms_values_a))
    time = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)

    # 绘制RMS曲线
    plt.figure(figsize=(14, 5))
    plt.plot(time, rms_values_a, label='RMS Value')
    plt.title('RMS Curve of Audio Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('RMS Value')
    plt.legend()
    plt.savefig(f'./rms_audio{i}.png')


    # 对齐长度
    min_length = min(len(rms_values_v), len(rms_values_a))
    rms_values_v = rms_values_v[:min_length]
    rms_values_a = rms_values_a[:min_length]

    # 计算皮尔逊相关系数
    pearson_corr = np.corrcoef(rms_values_v, rms_values_a)[0, 1]
    print(f"{i}: Pearson Correlation Coefficient: {pearson_corr}")

    # 计算互相关
    cross_corr = correlate(rms_values_v, rms_values_a, mode='full')

    # 绘制互相关图
    lags = np.arange(-len(rms_values_v) + 1, len(rms_values_a))
    plt.figure(figsize=(14, 5))
    plt.plot(lags, cross_corr, label='Cross-Correlation')
    plt.title('Cross-Correlation of Video and Audio RMS Curves')
    plt.xlabel('Lag (frames)')
    plt.ylabel('Cross-Correlation Value')
    plt.legend()
    plt.savefig(f'./rms_correlate{i}.png')