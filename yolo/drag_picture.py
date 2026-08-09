import gradio as gr
from ultralytics import YOLO

# 1. 加载 YOLO11 模型（自动使用你的 GPU）
model = YOLO("yolo11n.pt") # 也可以换成 yolo11s.pt 或 yolo11m.pt

# 2. 定义预测函数
def predict_image(input_image):
    if input_image is None:
        return None
    
    # 传入图片进行检测
    results = model(input_image)
    
    # results[0].plot() 会在图像上绘制边界框 (BGR 格式)
    # 转为 RGB 格式以在网页中正确显示颜色
    annotated_img = results[0].plot()[:, :, ::-1]
    
    return annotated_img

# 3. 构建 Gradio 拖拽界面
demo = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="numpy", label="把图片拖拽到这里 (或点击上传/粘贴)"),
    outputs=gr.Image(label="YOLO11 检测结果"),
    title="🎯 YOLO11 实时拖拽目标检测",
    description="支持直接将电脑里的任意图片拖入左侧区域，毫秒级输出检测结果。"
)

if __name__ == "__main__":
    # 启动本地服务，自动打开浏览器窗口
    demo.launch(inbrowser=True)