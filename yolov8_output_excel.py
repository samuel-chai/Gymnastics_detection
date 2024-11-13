import onnxruntime as ort
import numpy as np
import cv2
import time
import sys
import io
import tkinter as tk
import os
import pandas as pd
from tkinter import filedialog

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from utils import Keypoint
model_path = 'weights/yolov8x-pose.onnx'
keydet = Keypoint(model_path)   

def save_keypoints_to_excel(keypoints, output_file_path):
    """
    Saves keypoints to an Excel file.
    Each frame's keypoints are saved as rows, with x and y coordinates.
    """
    all_data = []
    for frame_id, frame_keypoints in enumerate(keypoints):
        if frame_keypoints[1] is not None:
            for person_keypoints in frame_keypoints[1][0]:  # Assuming one person per frame
                row = {'Frame': frame_id + 1}
                for kpt_id, kpt in enumerate(person_keypoints):
                    row[f'Keypoint_{kpt_id + 1}_x'] = kpt[0]  # x coordinate
                    row[f'Keypoint_{kpt_id + 1}_y'] = kpt[1]  # y coordinate
                all_data.append(row)

    df = pd.DataFrame(all_data)
    excel_file_path = output_file_path.replace('.npz', '_keypoints.xlsx')
    df.to_excel(excel_file_path, index=False)
    print(f"Keypoints saved to Excel file at {excel_file_path}")

if __name__ == '__main__':
    root = tk.Tk()
    root.withdraw()
    input_video_path = filedialog.askopenfilename(title="選擇視頻文件", filetypes=[("Video files", "*.mp4;*.avi;*.mov;*.mkv")])
    if not input_video_path:
        print("Error: No video file selected.")
        exit()
        
    video_folder = os.path.dirname(input_video_path)
    video_name = os.path.basename(input_video_path)


    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    all_keypoints = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        _, keypoints = keydet.inference(frame)
        all_keypoints.append(keypoints)
        frame_count += 1
        if frame_count % 10 == 0:
            print(f"已處理 {frame_count} 幀")
    
        

    save_keypoints_to_excel(all_keypoints, input_video_path)

    cap.release()
    print(f"檢測結果已保存至 {input_video_path.replace('.mp4', '_keypoints.xlsx')}")


