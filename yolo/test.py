from ultralytics import YOLO

# 1. 加载 YOLO11 最新预训练模型 (自动下载 yolo11n.pt)
model = YOLO("yolo11n.pt") 

# 2. 对图片进行检测并自动保存结果
results = model("https://ultralytics.com/images/bus.jpg", save=True)

# 3. 实时摄像头检测（在 GPU 加速下画面会极其流畅）
# model.predict(source="0", show=True)