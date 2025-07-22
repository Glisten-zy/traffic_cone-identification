import os
import json
from PIL import Image

def yolo_to_blip_objects(yolo_txt_path, img_width, img_height):
    objects = []
    with open(yolo_txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue  # 跳过格式错误的行
            class_id, x_c, y_c, w, h = map(float, parts)
            x = (x_c - w / 2) * img_width
            y = (y_c - h / 2) * img_height
            width = w * img_width
            height = h * img_height
            objects.append({
                "x": int(x),
                "y": int(y),
                "width": int(width),
                "height": int(height)
            })
    return objects

# 示例路径，请根据你的实际路径修改
image_dir = r"E:\rubbercone-captioning\images"       # 图像文件夹
label_dir = r"E:\rubbercone-captioning\labels"       # YOLO .txt 文件所在
output_jsonl = r"E:\rubbercone-captioning\outputs\train_blip_from_yolo.jsonl"

with open(output_jsonl, "w", encoding="utf-8") as fout:
    for filename in os.listdir(label_dir):
        if filename.endswith(".txt"):
            image_name = filename.replace(".txt", ".jpg")
            image_path = os.path.join(image_dir, image_name)
            label_path = os.path.join(label_dir, filename)

            if not os.path.exists(image_path):
                continue  # 如果图片不存在就跳过

            with Image.open(image_path) as img:
                width, height = img.size

            objects = yolo_to_blip_objects(label_path, width, height)

            entry = {
                "image": image_name,
                "objects": objects
            }
            fout.write(json.dumps(entry) + "\n")

output_jsonl  # 返回生成的 jsonl 文件路径

