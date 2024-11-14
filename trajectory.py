import onnxruntime as ort
import numpy as np
import cv2
import time
import tkinter as tk
from tkinter import filedialog, simpledialog

providers = [
    ('CUDAExecutionProvider', {'device_id': 0}),  # 可選 GPU 設備
    'CPUExecutionProvider',  # 或使用 CPU
]

# 定义人体关键点
KEYPOINT_NAMES = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
                  "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip",
                  "left_knee", "right_knee", "left_ankle", "right_ankle"]

# 选择特定关键点的索引
def select_keypoint():
    root = tk.Tk()
    root.withdraw()
    selected_keypoint = simpledialog.askstring("选择关键点", f"请输入关键点名称（{', '.join(KEYPOINT_NAMES)}）:")
    if selected_keypoint not in KEYPOINT_NAMES:
        print("Error: Invalid keypoint selected.")
        exit()
    return KEYPOINT_NAMES.index(selected_keypoint), selected_keypoint

# 重新定义模型与方法
class Keypoint():
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.label_name = self.session.get_outputs()[0].name

    def inference(self, image):
        img, (dw, dh) = letterbox(image)
        data = pre_process(img)

        pred = self.session.run([self.label_name], {self.input_name: data.astype(np.float32)})[0]
        pred = pred[0].transpose(1, 0)
        conf = 0.7
        pred = pred[pred[:, 4] > conf]

        if len(pred) == 0:
            print("没有检测到任何关键点")
            return image, None
        kpts = scale_keypoints(pred[0, 5:], min(img.shape[0] / image.shape[0], img.shape[1] / image.shape[1]), dw, dh)
        return image, kpts

# 一些预处理和辅助方法
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), scaleup=True):
    shape = im.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    im = cv2.copyMakeBorder(im, int(round(dh - 0.1)), int(round(dh + 0.1)),
                            int(round(dw - 0.1)), int(round(dw + 0.1)),
                            cv2.BORDER_CONSTANT, value=color)
    return im, (dw, dh)

def pre_process(img):
    img = img / 255.0
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, axis=0)

def scale_keypoints(kpts, gain, dw, dh):
    kpts[0::3] = (kpts[0::3] - dw) / gain
    kpts[1::3] = (kpts[1::3] - dh) / gain
    return kpts

# 主程序
if __name__ == '__main__':
    keypoint_model_path = 'weights/yolov8x-pose.onnx'
    keydet = Keypoint(keypoint_model_path)

    # 视频文件选择
    root = tk.Tk()
    root.withdraw()
    input_video_path = filedialog.askopenfilename(title="选择视频文件", filetypes=[("Video files", "*.mp4;*.avi;*.mov;*.mkv")])
    if not input_video_path:
        print("Error: No video file selected.")
        exit()

    output_video_path = filedialog.asksaveasfilename(title="选择保存视频文件位置", defaultextension=".mp4",
                                                     filetypes=[("MP4 files", "*.mp4")])
    if not output_video_path:
        print("Error: No save location selected.")
        exit()

    # 获取用户选择的关键点
    keypoint_idx, keypoint_name = select_keypoint()
    print(f"已选择关键点: {keypoint_name}")

    # 打开视频文件
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # 轨迹点存储
    trajectory_points = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Info: End of video file.")
            break

        # 对当前帧进行关键点检测
        frame, kpts = keydet.inference(frame)

        if kpts is not None:
            x, y, conf = kpts[keypoint_idx * 3], kpts[keypoint_idx * 3 + 1], kpts[keypoint_idx * 3 + 2]
            if conf > 0.5:
                trajectory_points.append((int(x), int(y)))  # 添加轨迹点
                for pt in trajectory_points:
                    cv2.circle(frame, pt, 3, (0, 255, 0), -1)  # 绘制轨迹点
                cv2.putText(frame, f"Tracking {keypoint_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # 实时显示处理后的视频帧
        cv2.imshow("Output Video", frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()
