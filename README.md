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

## 代碼解說
#### main_detection.py
1. 主要功能是可以同時檢測人體和跳馬台的比例
2. 可以輸出幾個主要csv
#### handle_scale.py
1. 在main基礎上另外新增可以暫停的功能, 按下space暫停
2. 另外在人出來後, 點畫面上的任意點, 可以計算出跟人剛出來(hip座標)的原點相差的距離(m)
3. 點擊的點也會顯示在畫面中