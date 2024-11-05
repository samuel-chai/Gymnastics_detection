import onnxruntime as ort
import numpy as np
import cv2
import time
import sys
import io
import tkinter as tk
import os
from tkinter import filedialog

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


# 定义一个调色板数组，其中每个元素是一个包含RGB值的列表，用于表示不同的颜色
palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                    [230, 230, 0], [255, 153, 255], [153, 204, 255],
                    [255, 102, 255], [255, 51, 255], [102, 178, 255],
                    [51, 153, 255], [255, 153, 153], [255, 102, 102],
                    [255, 51, 51], [153, 255, 153], [102, 255, 102],
                    [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                    [255, 255, 255]])
# 定义人体17个关键点的连接顺序，每个子列表包含两个数字，代表要连接的关键点的索引, 1鼻子 2左眼 3右眼 4左耳 5右耳 6左肩 7右肩 8左肘 9右肘 10左手腕 11右手腕 12左髋 13右髋 14左膝 15右膝 16左踝 17右踝
skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
            [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
            [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
# 通过索引从调色板中选择颜色，用于绘制人体骨架的线条，每个索引对应一种颜色
pose_limb_color = palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
# 通过索引从调色板中选择颜色，用于绘制人体的关键点，每个索引对应一种颜色
pose_kpt_color = palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]

providers = [
    ('CUDAExecutionProvider', {
        'device_id': 0,  # 可以选择GPU设备ID，如果你有多个GPU
    }),
    'CPUExecutionProvider',  # 也可以设置CPU作为备选
]

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), scaleup=True):
    '''  调整图像大小和两边灰条填充  '''
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    # 缩放比例 (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    # 只进行下采样 因为上采样会让图片模糊
    if not scaleup:
        r = min(r, 1.0)
    # 计算pad长宽
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))  # 保证缩放后图像比例不变
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    # 在较小边的两侧进行pad, 而不是在一侧pad
    dw /= 2
    dh /= 2
    # 将原图resize到new_unpad（长边相同，比例相同的新图）
    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    # 计算上下两侧的padding
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    # 计算左右两侧的padding
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    # 添加灰条
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im

def pre_process(img):
    # 归一化：将图像数据从0~255缩放到0~1之间，这一步是为了让模型更容易学习。
    img = img / 255.
    # 调整通道顺序：将图像从高度x宽度x通道数（H, W, C）调整为通道数x高度x宽度（C, H, W）的形式。
    # 这样做是因为许多深度学习框架要求输入的通道数在前。
    img = np.transpose(img, (2, 0, 1))
    # 增加一个维度：在0轴（即最前面）增加一个维度，将图像的形状从（C, H, W）变为（1, C, H, W）。
    # 这一步是为了满足深度学习模型输入时需要的批量大小（batch size）的维度，即使这里的批量大小为1。
    data = np.expand_dims(img, axis=0)
    return data


def xywh2xyxy(x):
    ''' 中心坐标、w、h ------>>> 左上点，右下点 '''
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def nms(dets, iou_thresh):
    # dets: N * M, N是bbox的个数，M的前4位是对应的 左上点，右下点
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)  # 求每个bbox的面积
    order = scores.argsort()[::-1]  # 对分数进行倒排序
    keep = []  # 用来保存最后留下来的bboxx下标
    while order.size > 0:
        i = order[0]  # 无条件保留每次迭代中置信度最高的bbox
        keep.append(i)
        # 计算置信度最高的bbox和其他剩下bbox之间的交叉区域
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        # 计算置信度高的bbox和其他剩下bbox之间交叉区域的面积
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        # 求交叉区域的面积占两者（置信度高的bbox和其他bbox）面积和的必烈
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        # 保留ovr小于thresh的bbox，进入下一次迭代。
        inds = np.where(ovr <= iou_thresh)[0]
        # 因为ovr中的索引不包括order[0]所以要向后移动一位
        order = order[inds + 1]
    output = []
    for i in keep:
        output.append(dets[i].tolist())
    return np.array(output)

def xyxy2xywh(a):
    ''' 左上点 右下点 ------>>> 左上点 宽 高 '''
    b = np.copy(a)
    b[:, 2] = a[:, 2] - a[:, 0]  # w
    b[:, 3] = a[:, 3] - a[:, 1]  # h
    return b

def scale_boxes(img1_shape, boxes, img0_shape):
    '''   将预测的坐标信息转换回原图尺度
    :param img1_shape: 缩放后的图像尺度
    :param boxes:  预测的box信息
    :param img0_shape: 原始图像尺度
    '''

    # 将检测框(x y w h)从img1_shape(预测图) 缩放到 img0_shape(原图)
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
    pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    boxes[:, 0] -= pad[0]
    boxes[:, 1] -= pad[1]
    boxes[:, :4] /= gain  # 检测框坐标点还原到原图上
    num_kpts = boxes.shape[1] // 3  # 56 // 3 = 18
    for kid in range(2, num_kpts + 1):
        boxes[:, kid * 3 - 1] = (boxes[:, kid * 3 - 1] - pad[0]) / gain
        boxes[:, kid * 3] = (boxes[:, kid * 3] - pad[1]) / gain
    # boxes[:, 5:] /= gain  # 关键点坐标还原到原图上
    clip_boxes(boxes, img0_shape)
    return boxes

def clip_boxes(boxes, shape):
    # 进行一个边界截断，以免溢出
    # 并且将检测框的坐标（左上角x，左上角y，宽度，高度）--->>>（左上角x，左上角y，右下角x，右下角y）
    top_left_x = boxes[:, 0].clip(0, shape[1])
    top_left_y = boxes[:, 1].clip(0, shape[0])
    bottom_right_x = (boxes[:, 0] + boxes[:, 2]).clip(0, shape[1])
    bottom_right_y = (boxes[:, 1] + boxes[:, 3]).clip(0, shape[0])
    boxes[:, 0] = top_left_x  # 左上
    boxes[:, 1] = top_left_y
    boxes[:, 2] = bottom_right_x  # 右下
    boxes[:, 3] = bottom_right_y

def plot_skeleton_kpts(im, kpts, steps=3):
    num_kpts = len(kpts) // steps  # 51 / 3 =17
    # 画点
    for kid in range(num_kpts):
        r, g, b = pose_kpt_color[kid]
        x_coord, y_coord = kpts[steps * kid], kpts[steps * kid + 1]
        conf = kpts[steps * kid + 2]
        if conf > 0.5:  # 关键点的置信度必须大于 0.5
            cv2.circle(im, (int(x_coord), int(y_coord)), 5, (int(r), int(g), int(b)), -1)
    # 画骨架
    for sk_id, sk in enumerate(skeleton):
        r, g, b = pose_limb_color[sk_id]
        pos1 = (int(kpts[(sk[0] - 1) * steps]), int(kpts[(sk[0] - 1) * steps + 1]))
        pos2 = (int(kpts[(sk[1] - 1) * steps]), int(kpts[(sk[1] - 1) * steps + 1]))
        conf1 = kpts[(sk[0] - 1) * steps + 2]
        conf2 = kpts[(sk[1] - 1) * steps + 2]
        if conf1 > 0.5 and conf2 > 0.5:  # 对于肢体，相连的两个关键点置信度 必须同时大于 0.5
            cv2.line(im, pos1, pos2, (int(r), int(g), int(b)), thickness=2)

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
        
        # 如果沒有檢測到物體，返回符合 Detectron2 格式的空結構
        if len(pred) == 0:
            return [np.empty((0, 5), dtype=np.float32)], [np.empty((0, 3), dtype=np.float32)]
        
        bboxs = xywh2xyxy(pred)
        bboxs = nms(bboxs, iou_thresh=0.6)
        bboxs = np.array(bboxs)
        bboxs = xyxy2xywh(bboxs)
        bboxs = scale_boxes(img.shape, bboxs, image.shape)
        
        # 構建符合 Detectron2 格式的嵌套結構，確保為 list([]) 結構並統一數據型別為 float32
        cls_boxes, cls_keyps = [np.empty((0, 5), dtype=np.float32)], [np.empty((0, 3), dtype=np.float32)]
        bboxes = []
        keypoints = []
        
        for box in bboxs:
            # Bounding box in Detectron2 format with confidence score
            det_bbox = [float(box[0]), float(box[1]), float(box[2]), float(box[3]), float(box[4])]
            bboxes.append(det_bbox)

            # Keypoints [(x, y, confidence), ...] in Detectron2 format
            kpts = box[5:]
            kpts_formatted = [[float(kpts[i]), float(kpts[i + 1]), float(kpts[i + 2])] if i + 2 < len(kpts) else [0.0, 0.0, 0.0] for i in range(0, 51, 3)]
            keypoints.append(kpts_formatted)
            
        # 將列表轉換為 numpy 陣列，確保 cls_boxes[1] 和 cls_keyps[1] 是統一的形狀
        cls_boxes[0] = np.array(bboxes, dtype=np.float32)
        cls_keyps[0] = np.array(keypoints, dtype=np.float32)

        return cls_boxes, cls_keyps
            
if __name__ == '__main__':
    model_path = 'weights/yolov8x-pose.onnx'
    keydet = Keypoint(model_path)

    root = tk.Tk()
    root.withdraw()
    input_video_path = filedialog.askopenfilename(title="選擇視頻文件", filetypes=[("Video files", "*.mp4;*.avi;*.mov;*.mkv")])
    if not input_video_path:
        print("Error: No video file selected.")
        exit()
        
    # 獲取影片所在的資料夾路徑
    video_folder = os.path.dirname(input_video_path)
    output_file_path = os.path.join(video_folder, "output_video_results.npz")

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

        # 每幀的檢測結果
        boxes, keypoints = keydet.inference(frame)
        all_boxes.append(boxes)
        all_keypoints.append(keypoints)

        frame_count += 1

    # 添加影片解析度的 metadata
    metadata = {"w": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "h": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
    
    # 保存成 npz 文件，符合 Detectron2 的嵌套結構
    np.savez(output_file_path, boxes=np.array(all_boxes, dtype=object), segments=np.array([[]]*len(all_boxes), dtype=object), keypoints=np.array(all_keypoints, dtype=object), metadata=metadata)

    cap.release()
    print(f"檢測結果已保存至 {output_file_path}")


