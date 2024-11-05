# object_detection.py
import onnxruntime as ort
from utils import letterbox, pre_process
import cv2
import numpy as np

class ObjectDetection:
    def __init__(self, model_path, providers):
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.label_name = self.session.get_outputs()[0].name

    def inference(self, image):
        img, (dw, dh) = letterbox(image)
        data = pre_process(img)
        pred = self.session.run([self.label_name], {self.input_name: data.astype(np.float32)})[0]
        pred = pred[0]
        pred = np.transpose(pred, (1, 0))
        conf = 0.5
        pred = pred[pred[:, 4] > conf]
        widths = []
        if len(pred) == 0:
            print("没有检测到任何物体")
            return image, widths
        else:
            bboxs = self.xywh2xyxy(pred)
            bboxs = self.nms(bboxs, iou_thresh=0.5)
            bboxs = np.array(bboxs)
            bboxs = self.scale_boxes(img.shape, bboxs, image.shape)
            for box in bboxs:
                x1, y1, x2, y2 = map(int, box[:4])
                width = x2 - x1
                widths.append((x1, y1, x2, y2, width))
                label = f"Object {box[4]:.2f}"
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            return image, widths

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

    def scale_boxes(self, img1_shape, boxes, img0_shape):
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
        boxes[:, [0, 2]] -= pad[0]
        boxes[:, [1, 3]] -= pad[1]
        boxes[:, :4] /= gain
        self.clip_boxes(boxes, img0_shape)
        return boxes

    def clip_boxes(self, boxes, shape):
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])
