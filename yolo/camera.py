from ultralytics import YOLO

# 加载轻量级模型
model = YOLO("yolov8n.pt")

# source=0 表示调用电脑默认摄像头；show=True 表示实时弹窗显示画面
# 按键盘上的 'q' 键可退出画面
model.predict(source="0", show=True)