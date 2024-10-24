import onnxruntime as ort
import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog
import os


# 定义一个调色板数组，其中每个元素是一个包含RGB值的列表，用于表示不同的颜色
palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                    [230, 230, 0], [255, 153, 255], [153, 204, 255],
                    [255, 102, 255], [255, 51, 255], [102, 178, 255],
                    [51, 153, 255], [255, 153, 153], [255, 102, 102],
                    [255, 51, 51], [153, 255, 153], [102, 255, 102],
                    [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                    [255, 255, 255]])

os.environ["CUDA_PATH"] = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.3"

providers = [
    ('CUDAExecutionProvider', {
        'device_id': 0,
    }),
    'CPUExecutionProvider'
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
        conf = 0.7
        pred = pred[pred[:, 4] > conf]
        if len(pred) == 0:
            print("没有检测到任何物体")
            return image
        else:
            # 中心宽高转左上点，右下点
            bboxs = xywh2xyxy(pred)
            # NMS处理
            bboxs = nms(bboxs, iou_thresh=0.6)
            # 坐标从左上点，右下点 到 左上点，宽，高.
            bboxs = np.array(bboxs)
            bboxs = scale_boxes(img.shape, bboxs, image.shape)
            # 画框
            for box in bboxs:
                x1, y1, x2, y2 = map(int, box[:4])
                label = f"Object {box[4]:.2f}"
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            return image

if __name__ == '__main__':
    model_path = 'weights/yolo8_platform.onnx'
    # 实例化模型
    obj_det = ObjectDetection(model_path)

    # 创建一个隐藏的主窗口
    root = tk.Tk()
    root.withdraw()

    # 弹出文件选择对话框选择输入视频文件
    input_video_path = filedialog.askopenfilename(title="选择视频文件", filetypes=[("Video files", "*.mp4;*.avi;*.mov;*.mkv")])
    if not input_video_path:
        print("Error: No video file selected.")
        exit()

    # 打开视频文件
    cap = cv2.VideoCapture(input_video_path)
    # 检查视频是否成功打开
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Info: End of video file.")
            break

        # 对读入的帧进行对象检测
        output_image = obj_det.inference(frame)

        # 实时显示处理后的视频帧
        cv2.imshow("Output Video", output_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()