import cv2
import numpy as np

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
    # 归一化：将图像数据从0~255缩放到0~1之间
    img = img / 255.0
    # 调整通道顺序：将图像从(H, W, C)调整为(C, H, W)
    img = np.transpose(img, (2, 0, 1))
    # 增加一个维度：将图像的形状从(C, H, W)变为(1, C, H, W)
    data = np.expand_dims(img, axis=0)
    return data