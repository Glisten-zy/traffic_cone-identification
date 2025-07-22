import json

def position_to_location(x, y, width, height):
    horiz = 'left' if x < width / 3 else 'center' if x < width * 2 / 3 else 'right'
    vert = 'top' if y < height / 3 else 'middle' if y < height * 2 / 3 else 'bottom'
    return f"{vert}-{horiz}"

def count_to_phrase(n):
    if n == 0:
        return "no traffic cones"
    elif n == 1:
        return "a single traffic cone"
    elif n == 2:
        return "two traffic cones"
    elif n <= 4:
        return f"{n} traffic cones"
    else:
        return f"multiple traffic cones"

def generate_natural_caption(entry):
    image = entry["image"]
    if "objects" not in entry or not entry["objects"]:
        return f'The image "{image}" shows no traffic cones.'

    bboxes = entry["objects"]
    count = len(bboxes)

    phrases = []
    for box in bboxes:
        x, y, w, h = box["x"], box["y"], box["width"], box["height"]
        cx, cy = x + w / 2, y + h / 2
        location = position_to_location(cx, cy, width=640, height=480)
        phrases.append(f"one traffic cone is located in the {location.replace('-', ' ')}")

    joined = ", and ".join(phrases)
    return f'The image "{image}" shows {count_to_phrase(count)}. Specifically, {joined}.'

# 路径保持不变
input_path = r"E:\rubbercone-captioning\outputs\train_blip_from_yolo.jsonl"
output_path = r"E:\rubbercone-captioning\outputs\train_blip_natural.jsonl"

with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
    for line in fin:
        try:
            entry = json.loads(line.strip())
            if "image" not in entry:
                continue  # 跳过无效行
            new_caption = generate_natural_caption(entry)
            new_entry = {
                "image": entry["image"],
                "caption": new_caption
            }
            fout.write(json.dumps(new_entry) + "\n")
        except Exception as e:
            print(f"❌ 跳过异常数据: {e}")

print(f"✅ 自然语言版本已生成：{output_path}")
