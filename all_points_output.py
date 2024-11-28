import onnxruntime as ort
import numpy as np
import cv2
import time
import sys
import io
import tkinter as tk
from tkinter import filedialog
import pandas as pd
import xlsxwriter

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 定义一个调色板数组，其中每个元素是一个包含RGB值的列表，用于表示不同的颜色
palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                    [230, 230, 0], [255, 153, 255], [153, 204, 255],
                    [255, 102, 255], [255, 51, 255], [102, 178, 255],
                    [51, 153, 255], [255, 153, 153], [255, 102, 102],
                    [255, 51, 51], [153, 255, 153], [102, 255, 102],
                    [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                    [255, 255, 255]])

# 定义人体17个关键点的连接顺序，每个子列表包含两个数字，代表要连接的关键点的索引
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
    shape = im.shape[:2]  # 原始图像尺寸
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])  # 计算缩放比例
    if not scaleup:
        r = min(r, 1.0)

    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))  # 缩放后的尺寸
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2  # 左右填充
    dh /= 2  # 上下填充

    # 调整图像尺寸并添加填充
    im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    im = cv2.copyMakeBorder(im, int(round(dh - 0.1)), int(round(dh + 0.1)),
                            int(round(dw - 0.1)), int(round(dw + 0.1)),
                            cv2.BORDER_CONSTANT, value=color)
    return im, (dw, dh)

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
        # 求交叉区域的面积占两者（置信度高的bbox和其他bbox）面积和的比例
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        # 保留ovr小于thresh的bbox，进入下一次迭代。
        inds = np.where(ovr <= iou_thresh)[0]
        # 因为ovr中的索引不包括order[0]所以要向后移动一位
        order = order[inds + 1]
    output = []
    for i in keep:
        output.append(dets[i].tolist())
    return np.array(output)

def scale_boxes(img1_shape, boxes, img0_shape):
    '''   将预测的坐标信息转换回原图尺度
    :param img1_shape: 缩放后的图像尺度
    :param boxes:  预测的box信息
    :param img0_shape: 原始图像尺度
    '''

    # 将检测框(x y w h)从img1_shape(预测图) 缩放到 img0_shape(原图)
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
    pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    boxes[:, [0, 2]] -= pad[0]  # x padding
    boxes[:, [1, 3]] -= pad[1]  # y padding
    boxes[:, :4] /= gain  # 检测框坐标点还原到原图上
    clip_boxes(boxes, img0_shape)
    return boxes

def clip_boxes(boxes, shape):
    # 进行一个边界截断，以免溢出
    # 并且将检测框的坐标（左上角x，左上角y，宽度，高度）--->>>（左上角x，左上角y，右下角x，右下角y）
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y

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

def scale_keypoints(kpts, gain, dw, dh):
    """根据缩放比例和填充量，调整关键点的坐标。"""
    kpts[0::3] = (kpts[0::3] - dw) / gain  # x-coordinates
    kpts[1::3] = (kpts[1::3] - dh) / gain  # y-coordinates
    return kpts

class ObjectDetection():
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.label_name = self.session.get_outputs()[0].name

    def inference(self, image):
        img, (dw, dh) = letterbox(image)
        data = pre_process(img)
        # 预测输出
        pred = self.session.run([self.label_name], {self.input_name: data.astype(np.float32)})[0]
        pred = pred[0]
        pred = np.transpose(pred, (1, 0))
        # 置信度阈值过滤
        conf = 0.5
        pred = pred[pred[:, 4] > conf]
        widths = []  # 初始化 widths 变量
        if len(pred) == 0:
            print("没有检测到任何物体")
            return image, widths
        else:
            # 中心宽高转左上点，右下点
            bboxs = xywh2xyxy(pred)
            # NMS处理
            bboxs = nms(bboxs, iou_thresh=0.5)
            # 坐标从左上点，右下点 到 左上点，宽，高.
            bboxs = np.array(bboxs)
            bboxs = scale_boxes(img.shape, bboxs, image.shape)
            # 画框
            for box in bboxs:
                x1, y1, x2, y2 = map(int, box[:4])
                width = x2 - x1  # 计算宽度
                widths.append((x1, y1, x2, y2, width))  # 保存检测框和宽度
                label = f"Object {box[4]:.2f}"
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            return image, widths
        
def calculate_velocity(results):
    """
    根据关键点的坐标计算每个点的速度 (x 和 y 方向)，单位为 m/s。
    :param results: 原始帧数据，每帧的关键点坐标。
    :return: 每帧关键点速度的数据。
    """
    velocities = []

    for i in range(1, len(results)):
        prev_frame = results[i - 1]
        curr_frame = results[i]
        frame_id = curr_frame[0]  # 当前帧号
        time_interval = curr_frame[1] - prev_frame[1]  # 计算时间间隔（秒）

        # 初始化当前帧的速度数据
        velocity_row = [frame_id]
        for j in range(2, len(curr_frame), 2):  # 遍历每个关键点的 x, y 坐标
            prev_x, prev_y = prev_frame[j], prev_frame[j + 1]
            curr_x, curr_y = curr_frame[j], curr_frame[j + 1]
            
            # 如果关键点坐标无效，跳过
            if prev_x == 0 or prev_y == 0 or curr_x == 0 or curr_y == 0:
                velocity_row.extend([0, 0])
            else:
                # 计算速度
                vel_x = (curr_x - prev_x) / time_interval
                vel_y = (curr_y - prev_y) / time_interval
                velocity_row.extend([vel_x, vel_y])

        velocities.append(velocity_row)

    return velocities
# class Keypoint():
#     def __init__(self, model_path):
#         self.session = ort.InferenceSession(model_path, providers=providers)
#         self.input_name = self.session.get_inputs()[0].name
#         self.label_name = self.session.get_outputs()[0].name
#         self.hip_y_base = None  # 初始化基准点
#         self.platform_ratio = None  # 初始化 platform_ratio
#         self.hip_x_avg = None  # 初始化 hip_x_avg
#         self.hip_y_relative = None  # 初始化 hip_y_relative
#         self.nose_x = None  # 初始化 nose_x
#         self.nose_y = None  # 初始化 nose_y

#     def inference(self, image):        
#         img, (dw, dh) = letterbox(image)  # 缩放并填充图像
#         data = pre_process(img)  # 预处理数据

#         pred = self.session.run([self.label_name], {self.input_name: data.astype(np.float32)})[0]
#         pred = pred[0].transpose(1, 0)  # 调整输出形状
#         conf = 0.7  # 置信度阈值

#         pred = pred[pred[:, 4] > conf]
#         if len(pred) == 0:
#             return image, []

#         # 转换 bbox 格式并进行 NMS
#         bboxs = xywh2xyxy(pred)
#         bboxs = nms(bboxs, iou_thresh=0.6)
#         bboxs = np.array(bboxs)
#         gain = min(img.shape[0] / image.shape[0], img.shape[1] / image.shape[1])

#         # 初始化变量
#         all_kpts = []

#         # 绘制检测结果
#         for box in bboxs:
#             det_bbox, det_scores, kpts = box[:4], box[4], box[5:]
#             kpts = scale_keypoints(kpts, gain, dw, dh)  # 调整关键点坐标
            
#             # 提取关键点的坐标
#             nose_x, nose_y = kpts[0 * 3], kpts[0 * 3 + 1]  # 鼻子
#             shoulder_left_x, shoulder_left_y = kpts[5 * 3], kpts[5 * 3 + 1]  # 左肩
#             shoulder_right_x, shoulder_right_y = kpts[6 * 3], kpts[6 * 3 + 1]  # 右肩
#             elbow_left_x, elbow_left_y = kpts[7 * 3], kpts[7 * 3 + 1]  # 左肘
#             elbow_right_x, elbow_right_y = kpts[8 * 3], kpts[8 * 3 + 1]  # 右肘
#             wrist_left_x, wrist_left_y = kpts[9 * 3], kpts[9 * 3 + 1]  # 左腕
#             wrist_right_x, wrist_right_y = kpts[10 * 3], kpts[10 * 3 + 1]  # 右腕
#             hip_left_x, hip_left_y = kpts[11 * 3], kpts[11 * 3 + 1]  # 左髖
#             hip_right_x, hip_right_y = kpts[12 * 3], kpts[12 * 3 + 1]  # 右髖
#             knee_left_x, knee_left_y = kpts[13 * 3], kpts[13 * 3 + 1]  # 左膝
#             knee_right_x, knee_right_y = kpts[14 * 3], kpts[14 * 3 + 1]  # 右膝
#             ankle_left_x, ankle_left_y = kpts[15 * 3], kpts[15 * 3 + 1]  # 左踝
#             ankle_right_x, ankle_right_y = kpts[16 * 3], kpts[16 * 3 + 1]  # 右踝

#             if self.platform_ratio is not None:
#                 kpts[0::3] *= self.platform_ratio  # 調整x坐標
#                 kpts[1::3] *= self.platform_ratio  # 調整y坐標
#             all_kpts.append(kpts)

#             # 检查是否检测到所有四个关键点
#             if shoulder_right_y > 0.5 and shoulder_left_y > 0.5 and hip_right_y > 0.5 and hip_left_y > 0.5:
#                 # 计算髋部的平均坐标
#                 self.hip_x_avg = (hip_right_x + hip_left_x) / 2
#                 hip_y_avg = (hip_right_y + hip_left_y) / 2

#                 # 将原点设置在画面的右下角
#                 self.hip_x_avg = image.shape[1] - self.hip_x_avg  # x 坐标从右边缘开始计算
#                 hip_y_avg = image.shape[0] - hip_y_avg  # y 坐标从底部开始计算

#                 # 如果基准点未设置，则设置基准点
#                 if self.hip_y_base is None:
#                     self.hip_y_base = hip_y_avg

#                 # 计算相对于基准点的 y 坐标
#                 self.hip_y_relative = hip_y_avg - self.hip_y_base

#                 # 将髋部的 x 和 y 坐标乘以 platform_ratio
#                 if self.platform_ratio is not None:
#                     self.hip_x_avg *= self.platform_ratio
#                     self.hip_y_relative *= self.platform_ratio

#                 # 计算鼻子相对于髋部的初始位置的坐标
#                 self.nose_x = (image.shape[1] - nose_x) - self.hip_x_avg
#                 self.nose_y = (image.shape[0] - nose_y) - self.hip_y_base

#                 if self.platform_ratio is not None:
#                     self.nose_x *= self.platform_ratio
#                     self.nose_y *= self.platform_ratio

#                 # 计算其他关键点相对于髋部的初始位置的坐标
#                 keypoints = {
#                     "Nose": (nose_x, nose_y),
#                     "Left Shoulder": (shoulder_left_x, shoulder_left_y),
#                     "Right Shoulder": (shoulder_right_x, shoulder_right_y),
#                     "Left Elbow": (elbow_left_x, elbow_left_y),
#                     "Right Elbow": (elbow_right_x, elbow_right_y),
#                     "Left Wrist": (wrist_left_x, wrist_left_y),
#                     "Right Wrist": (wrist_right_x, wrist_right_y),
#                     "Left Knee": (knee_left_x, knee_left_y),
#                     "Right Knee": (knee_right_x, knee_right_y),
#                     "Left Ankle": (ankle_left_x, ankle_left_y),
#                     "Right Ankle": (ankle_right_x, ankle_right_y)
#                 }

#                 # 顯示hip的座標=
#                 if self.hip_x_avg is not None and self.hip_y_relative is not None:
#                     cv2.putText(image, f"Hip: ({self.hip_x_avg:.2f} m, {self.hip_y_relative:.2f} m)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#                 for key, (x, y) in keypoints.items():
#                     rel_x = (image.shape[1] - x) - self.hip_x_avg
#                     rel_y = (image.shape[0] - y) - self.hip_y_base
#                     if self.platform_ratio is not None:
#                         rel_x *= self.platform_ratio
#                         rel_y *= self.platform_ratio
#                     cv2.putText(image, f"{key}: ({rel_x:.2f} m, {rel_y:.2f} m)", (10, 90 + 30 * list(keypoints.keys()).index(key)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#             # 绘制骨架和关键点
#             plot_skeleton_kpts(image, kpts)

#             return image, all_kpts

class Keypoint():
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.label_name = self.session.get_outputs()[0].name
        self.hip_y_base = None  # 初始化基准点
        self.platform_ratio = None  # 初始化 platform_ratio
        self.hip_x_avg = None  # 初始化 hip_x_avg
        self.hip_y_relative = None  # 初始化 hip_y_relative

    def inference(self, image):        
        img, (dw, dh) = letterbox(image)  # 缩放并填充图像
        data = pre_process(img)  # 预处理数据

        pred = self.session.run([self.label_name], {self.input_name: data.astype(np.float32)})[0]
        pred = pred[0].transpose(1, 0)  # 调整输出形状
        conf = 0.7  # 置信度阈值

        pred = pred[pred[:, 4] > conf]
        if len(pred) == 0:
            return image, []

        # 转换 bbox 格式并进行 NMS
        bboxs = xywh2xyxy(pred)
        bboxs = nms(bboxs, iou_thresh=0.6)
        bboxs = np.array(bboxs)
        gain = min(img.shape[0] / image.shape[0], img.shape[1] / image.shape[1])

        # 初始化变量
        all_kpts = []

        # 绘制检测结果
        for box in bboxs:
            det_bbox, det_scores, kpts = box[:4], box[4], box[5:]
            kpts = scale_keypoints(kpts, gain, dw, dh)  # 调整关键点坐标
            
            # 提取关键点的坐标
            all_kpts.append(kpts)

        return image, all_kpts

if __name__ == '__main__':
    keypoint_model_path = 'weights/yolov8x-pose.onnx'
    object_model_path = 'weights/yolo8_platform_v8.onnx'
    # 实例化模型
    keydet = Keypoint(keypoint_model_path)
    obj_det = ObjectDetection(object_model_path)

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
    # 获取视频的宽度、高度和帧率
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 创建 VideoWriter 对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 MP4 编码
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # 初始化帧数计数器和起始时间
    frame_count = 0
    start_time = time.time()

    # 初始化宽度列表和平台比例
    widths = []
    platform_ratio = None

    # 初始化髋部坐标和原点
    hip_origin = None
    right_edge_x = frame_width
    results = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Info: End of video file.")
            break

        # 对读入的帧进行人体关键点检测
        frame, all_kpts = keydet.inference(frame)
        # 对读入的帧进行物体检测
        output_image, detected_widths = obj_det.inference(frame)

        # 计算帧数和秒数
        frame_count += 1
        elapsed_time = time.time() - start_time
        seconds = elapsed_time
        

        # 计算前100帧的平均宽度
        if frame_count <= 100:
            widths.extend([width for _, _, _, _, width in detected_widths])
            if frame_count == 100:
                average_width = sum(widths) / len(widths)
                platform_ratio = 1.25 / average_width
                keydet.platform_ratio = platform_ratio  # 设置 platform_ratio
                print(f"Platform ratio: {platform_ratio:.6f} m/px")
                
        # 处理关键点数据
        for kpts in all_kpts:
            # 提取关键点的坐标
            nose_x, nose_y = kpts[0 * 3], kpts[0 * 3 + 1]  # 鼻子
            shoulder_left_x, shoulder_left_y = kpts[5 * 3], kpts[5 * 3 + 1]  # 左肩
            shoulder_right_x, shoulder_right_y = kpts[6 * 3], kpts[6 * 3 + 1]  # 右肩
            elbow_left_x, elbow_left_y = kpts[7 * 3], kpts[7 * 3 + 1]  # 左肘
            elbow_right_x, elbow_right_y = kpts[8 * 3], kpts[8 * 3 + 1]  # 右肘
            wrist_left_x, wrist_left_y = kpts[9 * 3], kpts[9 * 3 + 1]  # 左腕
            wrist_right_x, wrist_right_y = kpts[10 * 3], kpts[10 * 3 + 1]  # 右腕
            hip_left_x, hip_left_y = kpts[11 * 3], kpts[11 * 3 + 1]  # 左髖
            hip_right_x, hip_right_y = kpts[12 * 3], kpts[12 * 3 + 1]  # 右髖
            knee_left_x, knee_left_y = kpts[13 * 3], kpts[13 * 3 + 1]  # 左膝
            knee_right_x, knee_right_y = kpts[14 * 3], kpts[14 * 3 + 1]  # 右膝
            ankle_left_x, ankle_left_y = kpts[15 * 3], kpts[15 * 3 + 1]  # 左踝
            ankle_right_x, ankle_right_y = kpts[16 * 3], kpts[16 * 3 + 1]  # 右踝

            if platform_ratio is not None:
                kpts[0::3] *= platform_ratio  # 調整x坐標
                kpts[1::3] *= platform_ratio  # 調整y坐標

            # 检查是否检测到所有四个关键点
            if shoulder_right_y > 0.5 and shoulder_left_y > 0.5 and hip_right_y > 0.5 and hip_left_y > 0.5:
                # 计算髋部的平均坐标
                hip_x_avg = (hip_right_x + hip_left_x) / 2
                hip_y_avg = (hip_right_y + hip_left_y) / 2

                # 将原点设置在画面的右下角
                hip_x_avg = frame.shape[1] - hip_x_avg  # x 坐标从右边缘开始计算
                hip_y_avg = frame.shape[0] - hip_y_avg  # y 坐标从底部开始计算

                # 如果基准点未设置，则设置基准点
                if keydet.hip_y_base is None:
                    keydet.hip_y_base = hip_y_avg

                # 计算相对于基准点的 y 坐标
                hip_y_relative = hip_y_avg - keydet.hip_y_base

                # 将髋部的 x 和 y 坐标乘以 platform_ratio
                if platform_ratio is not None:
                    hip_x_avg *= platform_ratio
                    hip_y_relative *= platform_ratio

                # 计算鼻子相对于髋部的初始位置的坐标
                nose_x_rel = (frame.shape[1] - nose_x) - hip_x_avg
                nose_y_rel = (frame.shape[0] - nose_y) - keydet.hip_y_base

                if platform_ratio is not None:
                    nose_x_rel *= platform_ratio
                    nose_y_rel *= platform_ratio

                # 计算其他关键点相对于髋部的初始位置的坐标
                keypoints = {
                    "Nose": (nose_x, nose_y),
                    "Left Shoulder": (shoulder_left_x, shoulder_left_y),
                    "Right Shoulder": (shoulder_right_x, shoulder_right_y),
                    "Left Elbow": (elbow_left_x, elbow_left_y),
                    "Right Elbow": (elbow_right_x, elbow_right_y),
                    "Left Wrist": (wrist_left_x, wrist_left_y),
                    "Right Wrist": (wrist_right_x, wrist_right_y),
                    "Left Knee": (knee_left_x, knee_left_y),
                    "Right Knee": (knee_right_x, knee_right_y),
                    "Left Ankle": (ankle_left_x, ankle_left_y),
                    "Right Ankle": (ankle_right_x, ankle_right_y)
                }

                # 顯示hip的座標
                if hip_x_avg is not None and hip_y_relative is not None:
                    cv2.putText(output_image, f"Hip: ({hip_x_avg:.2f} m, {hip_y_relative:.2f} m)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                for key, (x, y) in keypoints.items():
                    rel_x = (frame.shape[1] - x) - hip_x_avg
                    rel_y = (frame.shape[0] - y) - keydet.hip_y_base
                    if platform_ratio is not None:
                        rel_x *= platform_ratio
                        rel_y *= platform_ratio
                    cv2.putText(output_image, f"{key}: ({rel_x:.2f} m, {rel_y:.2f} m)", (10, 90 + 30 * list(keypoints.keys()).index(key)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # 绘制骨架和关键点
                plot_skeleton_kpts(output_image, kpts)

                # 保存髋部和其他关键点的相对坐标数据
                frame_data = [frame_count, seconds, hip_x_avg, hip_y_relative]
                for key in ["Nose", "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow", "Left Wrist", "Right Wrist", "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"]:
                    rel_x, rel_y = (frame.shape[1] - keypoints[key][0]) - hip_x_avg, (frame.shape[0] - keypoints[key][1]) - keydet.hip_y_base
                    if platform_ratio is not None:
                        rel_x *= platform_ratio
                        rel_y *= platform_ratio
                    frame_data.extend([rel_x, rel_y])
                results.append(frame_data)


        # 在帧的左下角绘制帧数和秒数
        text = f"Frames: {frame_count}, Seconds: {seconds:.3f}"
        cv2.putText(output_image, text, (10, frame_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
        # 提取并显示当前帧的宽度
        if detected_widths:
            current_width = detected_widths[0][4]  # 提取第一个检测框的宽度
            cv2.putText(output_image, f"Platform Width: {current_width} px", (10, frame_height - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 实时显示处理后的视频帧
        cv2.imshow("Output Video", output_image)

        # 写入帧到输出视频
        out.write(output_image) 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

 
    # 释放资源
    cap.release()
    out.release()  # 释放 VideoWriter 对象
    cv2.destroyAllWindows()

    # 提示用户选择保存位置
    output_csv_path = filedialog.asksaveasfilename(
        title="选择保存 Excel 文件位置", 
        defaultextension=".xlsx",  # 改为 .xlsx
        filetypes=[("Excel files", "*.xlsx"), ("All Files", "*.*")]
    )

    if output_csv_path:
        if not output_csv_path.endswith(".xlsx"):
            output_csv_path += ".xlsx"

        # 创建新的列名
        keypoint_names = ["hip", "nose", "left_shoulder", "right_shoulder", 
                          "left_elbow", "right_elbow", "left_wrist", "right_wrist", 
                          "left_knee", "right_knee", "left_ankle", "right_ankle"]
        coordinate_columns = ["Frame", "Seconds"] + [f"{name}_{axis}" for name in keypoint_names for axis in ["X", "Y"]]

        # 计算速度
        velocities = calculate_velocity(results)

        # 创建速度列名
        velocity_columns = ["Frame"] + [f"{name}_{axis}_velocity" for name in keypoint_names for axis in ["X", "Y"]]    

        with pd.ExcelWriter(output_csv_path, engine='xlsxwriter') as writer:
            # 保存髋部和其他关键点的相对坐标数据到第一个表
            coord_df = pd.DataFrame(results, columns=coordinate_columns)
            coord_df.to_excel(writer, sheet_name='Coordinates', index=False)

            # 保存速度数据到第二个表
            velocity_df = pd.DataFrame(velocities, columns=velocity_columns)
            velocity_df.to_excel(writer, sheet_name='Velocities', index=False)

        print(f"Excel 文件已保存至: {output_csv_path}")
    else:
        print("Error: No save location selected.")