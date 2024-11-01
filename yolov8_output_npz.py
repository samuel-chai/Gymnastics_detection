import onnxruntime as ort
import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), scaleup=True):
    shape = im.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    return cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

class Keypoint:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path, providers=[
            ('CUDAExecutionProvider', {'device_id': 0}),
            'CPUExecutionProvider'
        ])
        self.input_name = self.session.get_inputs()[0].name
        self.label_name = self.session.get_outputs()[0].name

    def inference(self, image):
        img = letterbox(image)
        data = np.expand_dims(np.transpose(img / 255.0, (2, 0, 1)), axis=0).astype(np.float32)
        pred = self.session.run([self.label_name], {self.input_name: data})[0]

        pred = pred[0].reshape(-1, 56)  
        if pred.shape[1] < 5:
            print("Error: Predictions do not contain enough elements.")
            return []

        pred = pred[pred[:, 4] > 0.7]
        if len(pred) == 0:
            return []

        bboxs = xywh2xyxy(pred[:, :4])
        bboxs = np.hstack([bboxs, pred[:, 4:5]])  
        bboxs = nms(bboxs, 0.6)

        keypoints = [box[5:] for box in bboxs]
        return keypoints

def xywh2xyxy(x):
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  
    y[:, 1] = x[:, 1] - x[:, 3] / 2  
    y[:, 2] = x[:, 0] + x[:, 2] / 2  
    y[:, 3] = x[:, 1] + x[:, 3] / 2  
    return y

def nms(dets, iou_thresh):
    x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= iou_thresh)[0]
        order = order[inds + 1]

    return dets[keep]

def select_file(title, filetypes):
    return filedialog.askopenfilename(title=title, filetypes=filetypes)

def save_npz(data, save_path):
    np.savez(save_path, **{f"frame_{i}": kp for i, kp in enumerate(data)})

if __name__ == '__main__':
    model_path = 'weights/yolov8x-pose.onnx'
    keydet = Keypoint(model_path)

    root = tk.Tk()
    root.withdraw()

    input_video_path = select_file("Select Video", [("Video files", "*.mp4;*.avi;*.mov;*.mkv")])
    if not input_video_path:
        print("No video selected. Exiting.")
        exit()

    keypoints_path = filedialog.asksaveasfilename(title="Save Keypoints", defaultextension=".npz", filetypes=[("NPZ files", "*.npz")])
    if not keypoints_path:
        print("No save location selected for keypoints. Exiting.")
        exit()

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    all_keypoints = []
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break

        keypoints = keydet.inference(frame)
        all_keypoints.append(keypoints)

    save_npz(all_keypoints, keypoints_path)

    cap.release()
    print("Keypoints saved successfully.")
