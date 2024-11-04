import onnxruntime as ort
import numpy as np
import cv2
import time
import sys
import io
import tkinter as tk
from tkinter import filedialog

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 調色板、骨架和 YOLOv8 設置的其餘部分與原始代碼一致
# ...[省略無改動部分]...

class Keypoint():
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.label_name = self.session.get_outputs()[0].name

    def inference(self, image):
        img = letterbox(image)
        data = pre_process(img)
        pred = self.session.run([self.label_name], {self.input_name: data.astype(np.float32)})[0]
        pred = np.transpose(pred[0], (1, 0))
        conf = 0.7
        pred = pred[pred[:, 4] > conf]
        if len(pred) == 0:
            return None, None, None
        bboxs = xywh2xyxy(pred)
        bboxs = nms(bboxs, iou_thresh=0.6)
        bboxs = np.array(bboxs)
        bboxs = xyxy2xywh(bboxs)
        bboxs = scale_boxes(img.shape, bboxs, image.shape)

        boxes, keypoints, segments = [], [], []
        for box in bboxs:
            det_bbox = box[0:2]
            kpts = box[5:]
            boxes.append(det_bbox)
            keypoints.append(kpts[:2])
            segments.append(None)  # 沒有分割，填入 None
        return np.array(boxes), np.array(keypoints), np.array(segments)

if __name__ == '__main__':
    model_path = 'weights/yolov8x-pose.onnx'
    keydet = Keypoint(model_path)

    root = tk.Tk()
    root.withdraw()
    input_video_path = filedialog.askopenfilename(title="選擇視頻文件", filetypes=[("Video files", "*.mp4;*.avi;*.mov;*.mkv")])
    if not input_video_path:
        print("Error: No video file selected.")
        exit()

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    all_boxes, all_keypoints, all_segments = [], [], []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        boxes, keypoints, segments = keydet.inference(frame)
        if boxes is not None:
            all_boxes.extend(boxes)
            all_keypoints.extend(keypoints)
            all_segments.extend(segments)

        frame_count += 1

    metadata = {"description": "YOLOv8 人體姿勢檢測輸出"}
    np.savez("output_video_results.npz", boxes=np.array(all_boxes), segments=np.array(all_segments), keypoints=np.array(all_keypoints), metadata=metadata)

    cap.release()
    print("檢測結果已保存至 output_video_results.npz")
