import numpy as np
import tkinter as tk
from tkinter import filedialog

# 創建一個隱藏的主窗口
root = tk.Tk()
root.withdraw()

# 打開檔案對話框選擇 .npz 檔案
file_path = filedialog.askopenfilename(title="選擇 .npz 文件", filetypes=[("NPZ files", "*.npz")])

if file_path:
    # 載入 .npz 檔案
    data = np.load(file_path, allow_pickle=True)

    # 查看其中包含的變數名稱
    print("Variables in .npz file:", data.files)

    # 查看每個變數的內容
    for var in data.files:
        print(f"\nContent of {var}:")
        print(data[var])
else:
    print("未選擇文件。")
