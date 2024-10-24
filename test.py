import onnx

# 加载模型
model = onnx.load("weights/yolo8_platform.onnx")

# 将模型转换为较低的 IR 版本
onnx.checker.check_model(model)
onnx.helper.strip_doc_string(model)
model.ir_version = 8  # 设置为较低的 IR 版本

# 保存转换后的模型
onnx.save(model, "weights/yolo8_platform_v8.onnx")