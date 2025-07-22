import os
import shutil
import random

# === 设置路径 ===
image_dir = r"E:rubbercone-captioning\images"
label_dir = r"E:rubbercone-captioning\labels"
output_dir = "../datasets/cone"
splits = ["train", "val", "test"]
split_ratio = [0.7, 0.2, 0.1]  # 可调比例

# === 创建目标结构 ===
for split in splits:
    os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)

# === 获取所有图像名（确保图片和标签都有） ===
images = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
images = [f for f in images if os.path.exists(os.path.join(label_dir, os.path.splitext(f)[0] + ".txt"))]

# === 随机划分 ===
random.shuffle(images)
n_total = len(images)
n_train = int(split_ratio[0] * n_total)
n_val = int(split_ratio[1] * n_total)

train_set = images[:n_train]
val_set = images[n_train:n_train + n_val]
test_set = images[n_train + n_val:]

split_map = {"train": train_set, "val": val_set, "test": test_set}

# === 拷贝文件 ===
for split, files in split_map.items():
    for fname in files:
        stem = os.path.splitext(fname)[0]
        shutil.copy(os.path.join(image_dir, fname), os.path.join(output_dir, "images", split, fname))
        shutil.copy(os.path.join(label_dir, stem + ".txt"), os.path.join(output_dir, "labels", split, stem + ".txt"))
    print(f"[✓] 拷贝 {split}: {len(files)} 张图片")

# === 生成 cone.yaml ===
yaml_path = os.path.join(output_dir, "cone.yaml")
with open(yaml_path, "w", encoding="utf-8") as f:
    f.write(f"""train: {os.path.abspath(output_dir)}/images/train
val: {os.path.abspath(output_dir)}/images/val
test: {os.path.abspath(output_dir)}/images/test

nc: 1
names: ['traffic_cone']
""")

print(f"[✓] 数据集划分完成，共 {n_total} 张图像")
print(f"[✓] cone.yaml 已生成：{yaml_path}")
