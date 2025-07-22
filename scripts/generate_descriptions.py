import json
import os

INPUT = r"E:rubbercone-captioning\outputs\descriptions.jsonl"
OUTPUT = r"E:rubbercone-captioning\outputs\train_blip.jsonl"
IMAGE_DIR = "images"

os.makedirs("outputs", exist_ok=True)

with open(INPUT, "r", encoding="utf-8") as fin, open(OUTPUT, "w", encoding="utf-8") as fout:
    for line in fin:
        entry = json.loads(line.strip())
        image_name = entry["image"]
        caption = entry["description"]

        # 构造新格式
        new_entry = {
            "image": os.path.join(IMAGE_DIR, image_name),
            "caption": caption
        }

        fout.write(json.dumps(new_entry, ensure_ascii=False) + "\n")

print(f"✅ 已生成训练集 JSONL：{OUTPUT}")
