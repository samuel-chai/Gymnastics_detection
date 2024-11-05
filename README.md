Gymnastics Detection
## 專案介紹
這個專案是基於 YOLOv8 模型進行的體操動作檢測系統。利用 ONNX 格式的模型來實時檢測運動員的動作，協助教練和運動員進行表現分析。

## 功能
- 支援 `.mp4` 和 `.avi` 影片檢測
- 基於 YOLOv8 模型進行姿勢檢測
- 使用 ONNX 加速模型推理
- 支援多段影片檢測並輸出結果

## 安裝
1. 克隆此專案：
   ```bash
   git clone https://github.com/samuel-chai/Gymnastics_detection.git
   cd Gymnastics_detection

2. 另外裝yolo8x-pose.onnx, yolo8_platform_v8.onnx放入weights裡面
3. 直接執行main_detection.py
4. 選取執行video, 輸出位置, detection完後輸出csv
