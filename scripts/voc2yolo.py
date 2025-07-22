import os
import xml.etree.ElementTree as ET
from PIL import Image

# === 配置路径 ===
ANNOT_DIR = r"E:rubbercone-captioning\annotations"
IMAGE_DIR = r"E:rubbercone-captioning\images"
LABEL_DIR = r"E:rubbercone-captioning\labels"
CLASS_NAME = "traffic_cone"

# === 类别映射（只有一个类，编号为0） ===
class_to_id = {CLASS_NAME: 0}

# === 创建输出目录 ===
os.makedirs(LABEL_DIR, exist_ok=True)

# === 转换函数 ===
def convert_annotation(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    filename = root.find("filename").text
    image_path = os.path.join(IMAGE_DIR, filename)

    # 检查图像是否存在
    if not os.path.exists(image_path):
        print(f"[跳过] 找不到图片：{filename}")
        return

    # 获取图像尺寸
    with Image.open(image_path) as img:
        w, h = img.size

    label_lines = []

    for obj in root.findall("object"):
        cls_name = obj.find("name").text
        if cls_name != CLASS_NAME:
            continue

        xml_box = obj.find("bndbox")
        xmin = float(xml_box.find("xmin").text)
        ymin = float(xml_box.find("ymin").text)
        xmax = float(xml_box.find("xmax").text)
        ymax = float(xml_box.find("ymax").text)

        # 转为 YOLO 格式：x_center, y_center, width, height (归一化)
        x_center = (xmin + xmax) / 2.0 / w
        y_center = (ymin + ymax) / 2.0 / h
        box_width = (xmax - xmin) / w
        box_height = (ymax - ymin) / h

        label_lines.append(f"0 {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")

    if label_lines:
        label_filename = os.path.splitext(filename)[0] + ".txt"
        with open(os.path.join(LABEL_DIR, label_filename), "w") as f:
            f.write("\n".join(label_lines))
        print(f"[✓] 转换完成：{filename}")
    else:
        print(f"[跳过] 没有 traffic_cone 标注：{filename}")

# === 主流程 ===
xml_files = [f for f in os.listdir(ANNOT_DIR) if f.endswith(".xml")]
for xml_file in xml_files:
    convert_annotation(os.path.join(ANNOT_DIR, xml_file))
