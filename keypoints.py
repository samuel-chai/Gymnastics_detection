import onnxruntime as ort
from utils import letterbox, pre_process
import cv2
import numpy as np

class Keypoint:
    def __init__(self, model_path, providers):
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.label_name = self.session.get_outputs()[0].name
        self.hip_y_base = None
        self.platform_ratio = None
        self.hip_x_avg = None
        self.hip_y_relative = None
        self.angle = None
        self.angle_thigh_body = None
        self.angle_knee_left = None
        self.angle_knee_right = None

    def inference(self, image):
        img, (dw, dh) = letterbox(image)
        data = pre_process(img)

        pred = self.session.run([self.label_name], {self.input_name: data.astype(np.float32)})[0]
        pred = pred[0].transpose(1, 0)
        conf = 0.7
        pred = pred[pred[:, 4] > conf]

        if len(pred) == 0:
            print("没有检测到任何关键点")
            return image

        # 转换 bbox 格式并进行 NMS (Non-Maximum Suppression)
        bboxs = self.xywh2xyxy(pred)
        bboxs = self.nms(bboxs, iou_thresh=0.6)
        bboxs = np.array(bboxs)
        gain = min(img.shape[0] / image.shape[0], img.shape[1] / image.shape[1])

        # 绘制检测结果
        for box in bboxs:
            det_bbox, det_scores, kpts = box[:4], box[4], box[5:]
            kpts = self.scale_keypoints(kpts, gain, dw, dh)
            self.extract_keypoints(kpts, image)

        return image

    def xywh2xyxy(self, x):
        y = np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    def nms(self, dets, iou_thresh):
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]
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
        return np.array([dets[i] for i in keep])

    def scale_keypoints(self, kpts, gain, dw, dh):
        kpts[0::3] = (kpts[0::3] - dw) / gain  # x-coordinates
        kpts[1::3] = (kpts[1::3] - dh) / gain  # y-coordinates
        return kpts

    def extract_keypoints(self, kpts, image):
        # 提取关键点并计算相应的值，例如髋部的平均坐标
        # 在图像上绘制骨架等
        pass