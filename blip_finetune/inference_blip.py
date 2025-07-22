from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

# === 修改为你模型保存路径 ===
model_path = r"E:\rubbercone-captioning\blip_finetuned\checkpoint-130"
image_path = r"E:\rubbercone-captioning\images\01cd713f4d105844b04b3dd3577e76df.jpg"  # 替换为你要预测的图像路径

# === 加载模型和处理器 ===
processor = BlipProcessor.from_pretrained(model_path)
model = BlipForConditionalGeneration.from_pretrained(model_path)
model.eval()

# === 加载图片 ===
image = Image.open(image_path).convert("RGB")
inputs = processor(images=image, return_tensors="pt")

# === 推理 ===
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=50)

# === 输出描述 ===
caption = processor.decode(out[0], skip_special_tokens=True)
print("📝 图像描述：", caption)
