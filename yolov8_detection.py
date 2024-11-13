import onnxruntime as ort
import numpy as np
import cv2
import time
import sys
import io
import tkinter as tk
from tkinter import filedialog

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from utils import Keypoint

            
if __name__ == '__main__':
    model_path = 'weights/yolov8x-pose.onnx'
    # 实例化模型
    keydet = Keypoint(model_path)

    # 创建一个隐藏的主窗口
    root = tk.Tk()
    root.withdraw()

    # 弹出文件选择对话框选择输入视频文件
    input_video_path = filedialog.askopenfilename(title="选择视频文件", filetypes=[("Video files", "*.mp4;*.avi;*.mov;*.mkv")])
    if not input_video_path:
        print("Error: No video file selected.")
        exit()

    # 弹出文件保存对话框选择输出视频文件
    output_video_path = filedialog.asksaveasfilename(title="选择保存视频文件位置", defaultextension=".mp4", filetypes=[("MP4 files", "*.mp4"), ("AVI files", "*.avi"), ("MOV files", "*.mov"), ("MKV files", "*.mkv")])
    if not output_video_path:
        print("Error: No save location selected.")
        exit()

    # 打开视频文件
    cap = cv2.VideoCapture(input_video_path)
    # 检查视频是否成功打开
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()
    # 读取视频的基本信息
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    # 定义视频编码器和创建VideoWriter对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 根据文件名后缀使用合适的编码器
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # 初始化帧数计数器和起始时间
    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Info: End of video file.")
            break

        # 对读入的帧进行对象检测
        output_image = keydet.inference(frame)

        # 计算并打印帧速率
        frame_count += 1
        end_time = time.time()
        elapsed_time = end_time - start_time
        if elapsed_time > 0:
            fps = frame_count / elapsed_time
            print(f"FPS: {fps:.2f}")

        # 将处理后的帧写入输出视频
        out.write(output_image)

        # （可选）实时显示处理后的视频帧
        cv2.imshow("Output Video", output_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()



