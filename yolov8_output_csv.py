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
from tqdm import tqdm

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from utils import pre_process, xywh2xyxy, nms, scale_keypoints

providers = [
    ('CUDAExecutionProvider', {
        'device_id': 0,  # 可以选择GPU设备ID，如果你有多个GPU
    }),
    'CPUExecutionProvider',  # 也可以设置CPU作为备选
]

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), scaleup=True):
    shape = im.shape[:2]  # 原始圖像尺寸
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])  # 計算縮放比例
    if not scaleup:
        r = min(r, 1.0)

    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))  # 縮放後的尺寸
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2  # 左右填充
    dh /= 2  # 上下填充

    # 調整圖像尺寸並添加填充
    im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    im = cv2.copyMakeBorder(im, int(round(dh - 0.1)), int(round(dh + 0.1)),
                            int(round(dw - 0.1)), int(round(dw + 0.1)),
                            cv2.BORDER_CONSTANT, value=color)
    return im, (dw, dh)
class Keypoint:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.label_name = self.session.get_outputs()[0].name

    def inference(self, image):
        img, (dw, dh) = letterbox(image)  # 縮放並填充圖像
        data = pre_process(img)  # 预处理数据

        pred = self.session.run([self.label_name], {self.input_name: data.astype(np.float32)})[0]
        pred = pred[0].transpose(1, 0)  # 调整输出形状
        conf = 0.3  # 降低置信度阈值

        pred = pred[pred[:, 4] > conf]
        if len(pred) == 0:
            return None

        # 转换 bbox 格式并进行 NMS
        bboxs = xywh2xyxy(pred)
        bboxs = nms(bboxs, iou_thresh=0.6)
        bboxs = np.array(bboxs)
        gain = min(img.shape[0] / image.shape[0], img.shape[1] / image.shape[1])

        # 提取鼻子关键点
        nose_points = []
        for box in bboxs:
            kpts = box[5:]  # 获取关键点坐标
            kpts = scale_keypoints(kpts, gain, dw, dh)  # 调整关键点坐标
            nose_x, nose_y = kpts[0 * 3], kpts[0 * 3 + 1]  # 0索引处为鼻子
            nose_points.append((nose_x, nose_y))
        
        return nose_points


def save_nose_keypoints_to_csv(results, output_file_path):
    """
    Saves only the 'Nose' keypoint (x, y) coordinates to a CSV file.
    """
    if not results:
        print("沒有任何鼻子關鍵點的檢測結果，無法生成 CSV。")
        return
    
    df = pd.DataFrame(results, columns=["Frame", "Nose_x", "Nose_y"])
    csv_file_path = output_file_path.replace('.mp4', '_nose_keypoints.csv')
    df.to_csv(csv_file_path, index=False)
    print(f"Nose keypoints saved to CSV file at {csv_file_path}")

if __name__ == '__main__':
    keypoint_model_path = 'weights/yolov8x-pose.onnx'
    keydet = Keypoint(keypoint_model_path)
    root = tk.Tk()
    root.withdraw()

    input_video_path = filedialog.askopenfilename(title="选择视频文件", filetypes=[("Video files", "*.mp4;*.avi;*.mov;*.mkv")])
    if not input_video_path:
        print("Error: No video file selected.")
        exit()

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    results = []

    # 自定義進度條顯示
    def update_progress(current_frame, total_frames):
        progress = int((current_frame / total_frames) * 20)  # 50個方塊長度的進度條
        sys.stdout.write(f"\rIn Progress: [{'#' * progress}{'.' * (20 - progress)}] {current_frame}/{total_frames} frames")
        sys.stdout.flush()

    current_frame = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        nose_points = keydet.inference(frame)
        if nose_points:
            for nose_x, nose_y in nose_points:
                results.append([current_frame, nose_x, nose_y])
        
        current_frame += 1
        update_progress(current_frame, frame_count)

    base_name = os.path.splitext(os.path.basename(input_video_path))[0]
    output_csv_path = os.path.join(os.path.dirname(input_video_path), f"{base_name}_nose_keypoints.csv")
    save_nose_keypoints_to_csv(results, output_csv_path)

    cap.release()
    cv2.destroyAllWindows()