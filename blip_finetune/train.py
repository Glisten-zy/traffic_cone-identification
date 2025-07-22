import os
import json
from PIL import Image

import torch
from datasets import Dataset
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    TrainingArguments,
    Trainer,
)

# ==== 路径配置 ====
jsonl_path = r"E:\rubbercone-captioning\outputs\train_blip_natural.jsonl"  # 每行包含 {image, caption}
image_folder = r"E:\rubbercone-captioning\images"                 # 存放图片的文件夹
model_checkpoint = "Salesforce/blip-image-captioning-base"  # 初始模型

# ==== 加载数据 ====
def load_data(jsonl_path):
    data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            image_path = os.path.join(image_folder, item["image"])
            if os.path.exists(image_path):
                data.append({
                    "image_path": image_path,
                    "caption": item["caption"]
                })
    return Dataset.from_list(data)

dataset = load_data(jsonl_path)

# ==== 初始化处理器和模型 ====
processor = BlipProcessor.from_pretrained(model_checkpoint)
model = BlipForConditionalGeneration.from_pretrained(model_checkpoint)

# ==== 转换函数 ====
def transform(example):
    image = Image.open(example["image_path"]).convert("RGB")
    inputs = processor(images=image, text=example["caption"], return_tensors="pt", padding="max_length", truncation=True, max_length=32)
    inputs["labels"] = inputs.input_ids
    return {k: v[0] for k, v in inputs.items()}

# ==== 应用预处理 ====
dataset = dataset.map(transform)

# ==== 设置训练参数 ====
training_args = TrainingArguments(
    output_dir="./blip_finetuned",
    per_device_train_batch_size=4,
    num_train_epochs=5,
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    remove_unused_columns=False,
    fp16=torch.cuda.is_available(),  # 自动判断是否用 GPU
    report_to="none"
)

# ==== 创建 Trainer ====
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=processor,
)

# ==== 启动训练 ====
if __name__ == "__main__":
    trainer.train()
