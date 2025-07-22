import os
import sys
import csv
import json
import torch
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

# === 添加 yolov5 模块路径 ===
yolov5_path = r"E:\rubbercone-captioning\yolov5"
sys.path.append(os.path.abspath(yolov5_path))

from utils.general import non_max_suppression, scale_coords
from utils.dataloaders import LoadImages
from models.common import DetectMultiBackend
from utils.torch_utils import select_device

from transformers import BlipProcessor, BlipForConditionalGeneration

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
    if "objects" not in entry or not entry["objects"]:
        return f'The image "{image}" shows no traffic cones.'

    bboxes = entry["objects"]
    count = len(bboxes)

    position_counts = {}
    for box in bboxes:
        x, y, w, h = box["x"], box["y"], box["width"], box["height"]
        cx, cy = x + w / 2, y + h / 2
        location = position_to_location(cx, cy, width=640, height=480)
        position_counts[location] = position_counts.get(location, 0) + 1

    phrases = []
    for loc, n in position_counts.items():
        phrase = f"{count_to_phrase(n)} in the {loc.replace('-', ' ')}"
        phrases.append(phrase)

    joined = ", and ".join(phrases)
    return f'The image "{image}" shows {count_to_phrase(count)}. Specifically, {joined}.'

# === 坐标格式转换 ===
def yolo_predictions_to_objects(preds, im0, img_shape):
    h, w, _ = im0.shape
    results = []
    for *xyxy, conf, cls in preds:
        x1, y1, x2, y2 = map(int, xyxy)
        results.append({
            "x": x1,
            "y": y1,
            "width": x2 - x1,
            "height": y2 - y1
        })
    return results

# === 画检测框并保存图像 ===
def draw_boxes(image_path, objects, save_path):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for box in objects:
        x, y, w, h = box["x"], box["y"], box["width"], box["height"]
        draw.rectangle([x, y, x+w, y+h], outline="red", width=3)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    image.save(save_path)

# === 主流程 ===
def run_pipeline(image_path, yolo_weights, blip_model_dir, output_dir, device='cpu'):
    os.makedirs(output_dir, exist_ok=True)
    vis_dir = os.path.join(output_dir, "vis")
    os.makedirs(vis_dir, exist_ok=True)

    device = select_device(device)
    model = DetectMultiBackend(yolo_weights, device=device)
    dataset = LoadImages(image_path, img_size=640)

    processor = BlipProcessor.from_pretrained(blip_model_dir)
    blip_model = BlipForConditionalGeneration.from_pretrained(blip_model_dir).to(device)

    jsonl_path = os.path.join(output_dir, "captions.jsonl")
    csv_path = os.path.join(output_dir, "captions.csv")

    jsonl_file = open(jsonl_path, "w", encoding="utf-8")
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.DictWriter(csv_file, fieldnames=["image", "caption_template", "caption_blip"])
    csv_writer.writeheader()

    for path, img, im0s, vid_cap, s in dataset:
        img_tensor = torch.from_numpy(img).to(device).float() / 255.0
        if img_tensor.ndimension() == 3:
            img_tensor = img_tensor.unsqueeze(0)

        pred = model(img_tensor, augment=False, visualize=False)
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)[0]

        image_name = os.path.basename(path)
        objects = []

        if pred is not None and len(pred):
            pred[:, :4] = scale_coords(img_tensor.shape[2:], pred[:, :4], im0s.shape).round()
            objects = yolo_predictions_to_objects(pred, im0s, img_tensor.shape)

        if not objects:
            print(f"🚫 {image_name} 没有检测到目标")
            continue

        # 构造 entry
        entry = {"image": image_name, "objects": objects}
        caption_template = generate_natural_caption(entry)


        # BLIP 生成
        # BLIP 生成（使用模板作为 prompt）
        image = Image.open(path).convert("RGB")
        prompt = (
                f"There are {len(objects)} traffic cones detected in this image. "
                f"{' '.join([f'One is located at the {position_to_location(o['x']+o['width']/2, o['y']+o['height']/2, 640, 480).replace('-', ' ')}.' for o in objects])} "
                "Please describe the image in natural language."
            )

        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
        output = blip_model.generate(
                **inputs,
                max_length=60,
                num_beams=5,
                no_repeat_ngram_size=4,  # 限制 4gram 重复
                repetition_penalty=1.3,  # 惩罚重复词
                length_penalty=0.9,
                early_stopping=True
            )

        caption_blip = processor.decode(output[0], skip_special_tokens=True)


        # 可视化保存
        save_vis_path = os.path.join(vis_dir, Path(image_name).stem + "_with_boxes.jpg")
        draw_boxes(path, objects, save_vis_path)

        # 保存结果
        record = {
            "image": image_name,
            "caption_template": caption_template,
            "caption_blip": caption_blip
        }
        jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        csv_writer.writerow(record)

        # 打印
        print(f"\n🖼 图像: {image_name}")
        print(f"📝 模板: {caption_template}")
        print(f"📝 BLIP: {caption_blip}")

    jsonl_file.close()
    csv_file.close()
    print(f"\n✅ 已保存 JSONL 至: {jsonl_path}")
    print(f"✅ 已保存 CSV 至: {csv_path}")
    print(f"✅ 可视化图片保存在: {vis_dir}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run YOLO + BLIP Captioning Pipeline")
    parser.add_argument('--image_path', type=str, required=True, help='Image path or folder')
    parser.add_argument('--yolo_weights', type=str, required=True, help='YOLOv5 weights path')
    parser.add_argument('--blip_dir', type=str, required=True, help='BLIP model directory')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Directory to save outputs')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device: cpu or cuda')

    args = parser.parse_args()

    run_pipeline(
        image_path=args.image_path,
        yolo_weights=args.yolo_weights,
        blip_model_dir=args.blip_dir,
        output_dir=args.output_dir,
        device=args.device
    )

