# main.py
import cv2
import time
import tkinter as tk
from tkinter import filedialog
import pandas as pd
from keypoints import Keypoint
from object_detection import ObjectDetection

if __name__ == '__main__':
    keypoint_model_path = 'weights/yolov8x-pose.onnx'
    object_model_path = 'weights/yolo8_platform_v8.onnx'
    providers = [
        ('CUDAExecutionProvider', {'device_id': 0}),
        'CPUExecutionProvider'
    ]

    # 实例化模型
    keydet = Keypoint(keypoint_model_path, providers)
    obj_det = ObjectDetection(object_model_path, providers)

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
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    # 获取视频的宽度、高度和帧率
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 创建 VideoWriter 对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    paused = False  # 用于跟踪视频是否暂停
    frame_count = 0
    start_time = time.time()

    widths = []  # 初始化宽度列表和平台比例
    platform_ratio = None
    results = []
    click_coords_list = []  # 初始化点击坐标列表

    def mouse_callback(event, x, y, flags, param):
        global click_coords_list
        if event == cv2.EVENT_LBUTTONDOWN and paused:
            click_coords_list.append((x, y))

    cv2.namedWindow("Output Video")
    cv2.setMouseCallback("Output Video", mouse_callback)

    while True:
        if not paused:
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
                    keydet.platform_ratio = platform_ratio
                    print(f"Platform ratio: {platform_ratio:.6f} m/px")

            # 在帧的左下角绘制帧数和秒数
            text = f"Frames: {frame_count}, Seconds: {seconds:.3f}"
            cv2.putText(output_image, text, (10, frame_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
            # 提取并显示当前帧的宽度
            if detected_widths:
                current_width = detected_widths[0][4]
                cv2.putText(output_image, f"Platform Width: {current_width} px", (10, frame_height - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 实时显示处理后的视频帧
            cv2.imshow("Output Video", output_image)
            out.write(output_image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
        
        # 计算并显示点击点与髋部坐标的距离
        if paused:
            for idx, click_coords in enumerate(click_coords_list):
                click_x, click_y = click_coords
                if keydet.hip_x_avg != 0 and keydet.hip_y_relative != 0:
                    hip_x, hip_y = keydet.hip_x_avg, keydet.hip_y_relative
                    distance_x = (click_x - hip_x) * keydet.platform_ratio
                    distance_y = (click_y - hip_y) * keydet.platform_ratio
                    distance_text = f"Click {idx+1}: X={distance_x:.2f} m, Y={distance_y:.2f} m"
                    cv2.putText(output_image, distance_text, (1509, 50 + 25 * idx), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
                # 在画面上标记点击的点并添加标签
                cv2.circle(output_image, (click_x, click_y), 5, (0, 0, 255), -1)
                cv2.putText(output_image, f"Click {idx+1}", (click_x + 10, click_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
            cv2.imshow("Output Video", output_image)

    # 結束时提示用户选择 CSV 文件的保存位置
    root = tk.Tk()
    root.withdraw()
    output_csv_path = filedialog.asksaveasfilename(
        title="选择保存 CSV 文件位置", 
        defaultextension=".csv", 
        filetypes=[("CSV files", "*.csv")]
    )
    
    if output_csv_path:
        df = pd.DataFrame(results, columns=["Frame", "Seconds", "Hip_X(m)", "Hip_Y(m)", "Angle(deg)", "Thigh_Body_Angle(deg)", "Left_Knee_Angle(deg)", "Right_Knee_Angle(deg)"])
        df.to_csv(output_csv_path, index=False)
        print(f"CSV 文件已保存至: {output_csv_path}")
    else:
        print("Error: No save location selected.")

    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()
