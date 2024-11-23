import onnxruntime as ort
import numpy as np
import cv2
import time
import sys
import io
import tkinter as tk
from tkinter import filedialog
import pandas as pd

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

class Keypoint():
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.label_name = self.session.get_outputs()[0].name
        self.hip_y_base = None  # 初始化基准点
        self.platform_ratio = None  # 初始化 platform_ratio
        self.hip_x_avg = None  # 初始化 hip_x_avg
        self.hip_y_relative = None  # 初始化 hip_y_relative
        self.angle = None
        self.angle_thigh_body = None  # 初始化 angle_thigh_body
        self.angle_knee_left = None  # 初始化 angle_knee_left
        self.angle_knee_right = None  # 初始化 angle_knee_right

    def inference(self, image):
        img, (dw, dh) = letterbox(image)  # 缩放并填充图像
        data = pre_process(img)  # 预处理数据

        pred = self.session.run([self.label_name], {self.input_name: data.astype(np.float32)})[0]
        pred = pred[0].transpose(1, 0)  # 调整输出形状
        conf = 0.7  # 置信度阈值

        pred = pred[pred[:, 4] > conf]
        if len(pred) == 0:
            print("没有检测到任何关键点")
            return image

        # 转换 bbox 格式并进行 NMS
        bboxs = xywh2xyxy(pred)
        bboxs = nms(bboxs, iou_thresh=0.6)
        bboxs = np.array(bboxs)
        gain = min(img.shape[0] / image.shape[0], img.shape[1] / image.shape[1])

        # 初始化变量
        

        # 绘制检测结果
        for box in bboxs:
            det_bbox, det_scores, kpts = box[:4], box[4], box[5:]
            kpts = scale_keypoints(kpts, gain, dw, dh)  # 调整关键点坐标
            
            # 提取左右肩和左右髋部的坐标
            shoulder_left_x, shoulder_left_y = kpts[5 * 3], kpts[5 * 3 + 1]  # 左肩
            shoulder_right_x, shoulder_right_y = kpts[6 * 3], kpts[6 * 3 + 1]  # 右肩
            hip_left_x, hip_left_y = kpts[11 * 3], kpts[11 * 3 + 1]  # 左髖
            hip_right_x, hip_right_y = kpts[12 * 3], kpts[12 * 3 + 1]  # 右髖
            
            # 提取左右膝和左右腳踝的座標
            knee_left_x, knee_left_y = kpts[13 * 3], kpts[13 * 3 + 1]
            knee_right_x, knee_right_y = kpts[14 * 3], kpts[14 * 3 + 1]
            ankle_left_x, ankle_left_y = kpts[15 * 3], kpts[15 * 3 + 1]
            ankle_right_x, ankle_right_y = kpts[16 * 3], kpts[16 * 3 + 1]

            # 提取左肘和左腕的坐标
            # 提取左肘和左腕的座標（COCO 格式的正確索引）
            elbow_left_x, elbow_left_y = kpts[7 * 3], kpts[7 * 3 + 1]  # 左肘
            wrist_left_x, wrist_left_y = kpts[9 * 3], kpts[9 * 3 + 1]  # 左腕

             # 检查是否检测到所有四个关键点
            if shoulder_right_y > 0.5 and shoulder_left_y > 0.5 and hip_right_y > 0.5 and hip_left_y > 0.5:
                # 计算髋部的平均坐标
                self.hip_x_avg = (hip_right_x + hip_left_x) / 2
                hip_y_avg = (hip_right_y + hip_left_y) / 2

                # 将原点设置在画面的右下角
                self.hip_x_avg = image.shape[1] - self.hip_x_avg  # x 坐标从右边缘开始计算
                hip_y_avg = image.shape[0] - hip_y_avg  # y 坐标从底部开始计算

                # 如果基准点未设置，则设置基准点
                if self.hip_y_base is None:
                    self.hip_y_base = hip_y_avg

                # 计算相对于基准点的 y 坐标
                self.hip_y_relative = hip_y_avg - self.hip_y_base

                # 将髋部的 x 和 y 坐标乘以 platform_ratio
                if self.platform_ratio is not None:
                    self.hip_x_avg *= self.platform_ratio
                    self.hip_y_relative *= self.platform_ratio
                            
            # 计算左肘和左腕之间的连线与水平线的夹角
            if elbow_left_x is not None and wrist_left_x is not None:
                dx = wrist_left_x - elbow_left_x  # 注意这里的方向
                dy = wrist_left_y - elbow_left_y  # 确保方向一致

                # 使用 arctan2 计算夹角，并将其转换到 0-180 度范围
                self.angle = np.degrees(np.arctan2(abs(dy), abs(dx)))  # 取绝对值保证方向一致

                # 确保结果在 0-180° 之间（只考虑前臂与水平线夹角的正值）
                if self.angle > 180:
                    self.angle = 360 - self.langle
                    
           # 計算左大腿直線(左膝和左髋部)和身體(左hip和左shoulder)連線間的夾角
            if knee_left_x is not None and hip_left_x is not None and shoulder_left_x is not None:
                # 向量1: 从左髋到左膝
                dx1 = knee_left_x - hip_left_x
                dy1 = knee_left_y - hip_left_y

                # 向量2: 从左髋到左肩
                dx2 = shoulder_left_x - hip_left_x
                dy2 = shoulder_left_y - hip_left_y

                # 计算向量的点积
                dot_product = dx1 * dx2 + dy1 * dy2

                # 计算向量的模长
                magnitude1 = np.sqrt(dx1**2 + dy1**2)
                magnitude2 = np.sqrt(dx2**2 + dy2**2)

                # 计算两个向量之间的夹角（弧度）
                angle_radians = np.arccos(dot_product / (magnitude1 * magnitude2))

                # 将弧度转换为角度
                angle_thigh_body = np.degrees(angle_radians)

                # 确保结果在 0-180° 之间
                if angle_thigh_body > 180:
                    angle_thigh_body = 360 - angle_thigh_body

                self.angle_thigh_body = angle_thigh_body

            # 計算膝蓋夾腳(hip到knee的連線與knee到ankle的連線), 左右都要計算
            # 左膝
            if hip_left_x is not None and knee_left_x is not None and ankle_left_x is not None:
                # 向量1: 从左膝到左髋
                dx1_left = hip_left_x - knee_left_x
                dy1_left = hip_left_y - knee_left_y
                # 向量2: 从左膝到左踝
                dx2_left = ankle_left_x - knee_left_x
                dy2_left = ankle_left_y - knee_left_y

                # 计算向量的点积
                dot_product_left = dx1_left * dx2_left + dy1_left * dy2_left

                # 计算向量的模长
                magnitude1_left = np.sqrt(dx1_left**2 + dy1_left**2)
                magnitude2_left = np.sqrt(dx2_left**2 + dy2_left**2)

                # 计算两个向量之间的夹角（弧度）
                angle_radians_left = np.arccos(dot_product_left / (magnitude1_left * magnitude2_left))

                # 将弧度转换为角度
                angle_knee_left = np.degrees(angle_radians_left)

                # 确保结果在 0-180° 之间
                if angle_knee_left > 180:
                    angle_knee_left = 360 - angle_knee_left

                self.angle_knee_left = angle_knee_left

            # 右膝
            if hip_right_x is not None and knee_right_x is not None and ankle_right_x is not None:
                # 向量1: 从右膝到右髋
                dx1_right = hip_right_x - knee_right_x
                dy1_right = hip_right_y - knee_right_y
                # 向量2: 从右膝到右踝
                dx2_right = ankle_right_x - knee_right_x
                dy2_right = ankle_right_y - knee_right_y

                # 计算向量的点积
                dot_product_right = dx1_right * dx2_right + dy1_right * dy2_right

                # 计算向量的模长
                magnitude1_right = np.sqrt(dx1_right**2 + dy1_right**2)
                magnitude2_right = np.sqrt(dx2_right**2 + dy2_right**2)

                # 计算两个向量之间的夹角（弧度）
                angle_radians_right = np.arccos(dot_product_right / (magnitude1_right * magnitude2_right))

                # 将弧度转换为角度
                angle_knee_right = np.degrees(angle_radians_right)

                # 确保结果在 0-180° 之间
                if angle_knee_right > 180:
                    angle_knee_right = 360 - angle_knee_right

                self.angle_knee_right = angle_knee_right



        # 在视频帧的左上角显示髋部的平均坐标
        if self.hip_x_avg is not None and self.hip_y_relative is not None:
            cv2.putText(image, f"Hip: ({self.hip_x_avg:.2f} m, {self.hip_y_relative:.2f} m)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 在视频帧的左上角显示左肘和左腕的夹角
        if self.angle is not None:
            cv2.putText(image, f"Angle between arm & platform: {self.angle:.2f} degrees", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
        # 在视频帧的左上角显示左大腿和身体的夹角
        if self.angle_thigh_body is not None:
            cv2.putText(image, f"Angle between thigh & body: {self.angle_thigh_body:.2f} degrees", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 在视频帧的左上角显示左膝和右膝的夹角
        if self.angle_knee_left is not None:
            cv2.putText(image, f"Left Knee Angle: {self.angle_knee_left:.2f} degrees", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if self.angle_knee_right is not None:
            cv2.putText(image, f"Right Knee Angle: {self.angle_knee_right:.2f} degrees", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 绘制骨架和关键点
        plot_skeleton_kpts(image, kpts)

        return image


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
        frame = keydet.inference(frame)
        # 对读入的帧进行物体检测
        output_image, detected_widths = obj_det.inference(frame)

        # 计算帧数和秒数
        frame_count += 1
        elapsed_time = time.time() - start_time
        seconds = elapsed_time

        # 提取髋部的平均座標和角度
        hip_x_avg = keydet.hip_x_avg if keydet.hip_x_avg is not None else 0
        hip_y_relative = keydet.hip_y_relative if keydet.hip_y_relative is not None else 0
        angle = keydet.angle if keydet.angle is not None else 0
        angle_thigh_body = keydet.angle_thigh_body if keydet.angle_thigh_body is not None else 0
        angle_knee_left = keydet.angle_knee_left if keydet.angle_knee_left is not None else 0
        angle_knee_right = keydet.angle_knee_right if keydet.angle_knee_right is not None else 0

        # 收集當前幀的數據
        results.append([frame_count, seconds, hip_x_avg, hip_y_relative, angle, angle_thigh_body, angle_knee_left, angle_knee_right])
        

        # 计算前100帧的平均宽度
        if frame_count <= 100:
            widths.extend([width for _, _, _, _, width in detected_widths])
            if frame_count == 100:
                average_width = sum(widths) / len(widths)
                platform_ratio = 1.25 / average_width
                keydet.platform_ratio = platform_ratio  # 设置 platform_ratio
                print(f"Platform ratio: {platform_ratio:.6f} m/px")

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

    # 結束時提示用戶選擇 CSV 文件的保存位置
    root = tk.Tk()
    root.withdraw()  # 隱藏主窗口
    output_csv_path = filedialog.asksaveasfilename(
    title="選擇保存 CSV 文件位置", 
    defaultextension=".csv", 
    filetypes=[("CSV files", "*.csv")]
    
)
    # 如果選擇了保存位置，將結果保存為 CSV
    # if output_csv_path:
    #     df = pd.DataFrame(results, columns=["Frame", "Seconds", "Hip_X(m)", "Hip_Y(m)", "Angle(deg)"])
    #     df.to_csv(output_csv_path, index=False)
    #     print(f"CSV 文件已保存至: {output_csv_path}")
    # else:
    #     print("Error: No save location selected.")
        
    if output_csv_path:
        df = pd.DataFrame(results, columns=["Frame", "Seconds", "Hip_X(m)", "Hip_Y(m)", "Angle(deg)", "Thigh_Body_Angle(deg)", "Left_Knee_Angle(deg)", "Right_Knee_Angle(deg)"])
        df.to_csv(output_csv_path, index=False)
        print(f"CSV 文件已保存至: {output_csv_path}")
    else:
        print("Error: No save location selected.")


    # 释放资源
    cap.release()
    out.release()  # 释放 VideoWriter 对象
    cv2.destroyAllWindows()