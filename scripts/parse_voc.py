import os
import json
import xml.etree.ElementTree as ET
from tqdm import tqdm

def parse_single_xml(xml_path):
    """
    解析单个 VOC XML 文件，提取目标并生成描述，同时保存目标坐标
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    image_name = root.find("filename").text
    objects = root.findall("object")

    boxes = []
    for obj in objects:
        name = obj.find("name").text
        if name != "traffic_cone":
            continue
        bndbox = obj.find("bndbox")
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)
        width = xmax - xmin
        height = ymax - ymin
        boxes.append({
            "x": xmin,
            "y": ymin,
            "width": width,
            "height": height
        })

    # 生成自然语言描述
    if len(boxes) == 0:
        description = f'The image "{image_name}" contains no visible traffic cones.'
    elif len(boxes) == 1:
        x, y = boxes[0]["x"], boxes[0]["y"]
        description = f'The image "{image_name}" contains one traffic cone near position ({x}, {y}).'
    else:
        description = f'The image "{image_name}" contains {len(boxes)} traffic cones.'

    return {
        "image": image_name,
        "objects": boxes,
        "caption": description
    }

def parse_voc_folder(xml_folder, output_path):
    """
    扫描文件夹下所有 XML，解析并写入 JSONL
    """
    all_data = []
    for file in tqdm(os.listdir(xml_folder)):
        if file.endswith(".xml"):
            xml_path = os.path.join(xml_folder, file)
            parsed = parse_single_xml(xml_path)
            all_data.append(parsed)

    with open(output_path, "w", encoding="utf-8") as f:
        for item in all_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"✅ 已保存 JSONL 到：{output_path}")

if __name__ == "__main__":
    xml_folder = r"E:\rubbercone-captioning\annotations"  # 替换为你的 XML 文件夹路径
    output_jsonl = r"E:\rubbercone-captioning\outputs\descriptions_object.jsonl"  # 替换为你想保存的路径
    parse_voc_folder(xml_folder, output_jsonl)
