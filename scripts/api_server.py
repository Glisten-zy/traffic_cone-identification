import os
import sys
import io
import base64
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image, ImageDraw
import torch
import numpy as np

# 添加 yolov5 模块路径
yolov5_path = r"E:\rubbercone-captioning\yolov5"
sys.path.append(os.path.abspath(yolov5_path))
from utils.general import non_max_suppression, scale_coords
from models.common import DetectMultiBackend
from utils.torch_utils import select_device

from transformers import BlipProcessor, BlipForConditionalGeneration

# 初始化模型
device = select_device("cuda" if torch.cuda.is_available() else "cpu")
yolo_weights = r"E:\rubbercone-captioning\yolov5\runs\train\cone_detect2\weights\best.pt"
blip_model_dir = r"E:\rubbercone-captioning\blip_finetuned\checkpoint-130"

yolo_model = DetectMultiBackend(yolo_weights, device=device)
blip_processor = BlipProcessor.from_pretrained(blip_model_dir)
blip_model = BlipForConditionalGeneration.from_pretrained(blip_model_dir).to(device)

# === 模板生成辅助函数 ===
def position_to_location(x, y, width, height):
    horiz = 'left' if x < width / 3 else 'center' if x < width * 2 / 3 else 'right'
    vert = 'top' if y < height / 3 else 'middle' if y < height * 2 / 3 else 'bottom'
    return f"{vert}-{horiz}"

def count_to_phrase(n):
    return {
        0: "no traffic cones",
        1: "a single traffic cone",
        2: "two traffic cones"
    }.get(n, f"{n} traffic cones" if n <= 4 else "multiple traffic cones")

def generate_natural_caption(entry):
    image = entry["image"]
    bboxes = entry["objects"]
    count = len(bboxes)
    if count == 0:
        return f'The image "{image}" shows no traffic cones.'

    position_counts = {}
    for box in bboxes:
        x, y, w, h = box["x"], box["y"], box["width"], box["height"]
        cx, cy = x + w / 2, y + h / 2
        location = position_to_location(cx, cy, width=640, height=480)
        position_counts[location] = position_counts.get(location, 0) + 1

    phrases = [f"{count_to_phrase(n)} in the {loc.replace('-', ' ')}" for loc, n in position_counts.items()]
    joined = ", and ".join(phrases)
    return f'The image "{image}" shows {count_to_phrase(count)}. Specifically, {joined}.'

# === 推理主函数 ===
def run_inference(image_bytes, image_name):
    im0 = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    im = im0.resize((640, 640))
    img_tensor = torch.from_numpy(np.array(im)).permute(2, 0, 1).float().to(device) / 255.0
    img_tensor = img_tensor.unsqueeze(0)

    with torch.no_grad():
        pred = yolo_model(img_tensor)
        pred = non_max_suppression(pred, 0.25, 0.45)[0]

    objects = []
    if pred is not None and len(pred):
        pred[:, :4] = scale_coords(img_tensor.shape[2:], pred[:, :4], im0.size[::-1]).round()
        for *xyxy, conf, cls in pred:
            x1, y1, x2, y2 = map(int, xyxy)
            objects.append({"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1})

    draw = ImageDraw.Draw(im0)
    for box in objects:
        x, y, w, h = box["x"], box["y"], box["width"], box["height"]
        draw.rectangle([x, y, x + w, y + h], outline="red", width=3)

    # 文本模板
    entry = {"image": image_name, "objects": objects}
    caption_template = generate_natural_caption(entry)

    # BLIP 推理
    try:
        inputs = blip_processor(im0, "A photo of a traffic cone scene", return_tensors="pt").to(device)
        output = blip_model.generate(**inputs, max_length=50)
        caption_blip = blip_processor.decode(output[0], skip_special_tokens=True)
    except:
        caption_blip = "Error in BLIP generation"

    # 转 base64 图片
    buffered = io.BytesIO()
    im0.save(buffered, format="JPEG")
    vis_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    vis_dataurl = f"data:image/jpeg;base64,{vis_b64}"

    return {
        "image_name": image_name,
        "caption_template": caption_template,
        "caption_blip": caption_blip,
        "vis_image_base64": vis_dataurl
    }

# === FastAPI 接口 ===
app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    result = run_inference(image_bytes, file.filename)
    return JSONResponse(content=result)
