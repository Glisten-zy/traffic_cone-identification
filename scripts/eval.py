import json
from pathlib import Path
from collections import defaultdict
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge import Rouge
from tqdm import tqdm

# === 读取 jsonl 文件 ===
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

# === 评估函数 ===
def evaluate(gold_file, pred_file):
    gold_data = load_jsonl(gold_file)
    pred_data = load_jsonl(pred_file)

    gold_dict = {entry["image"]: entry["caption"] for entry in gold_data}
    pred_dict = {entry["image"]: entry["caption_blip"] for entry in pred_data}

    refs, hyps = [], []
    matched = 0

    for image in tqdm(pred_dict):
        if image in gold_dict:
            ref = gold_dict[image].strip().lower()
            hyp = pred_dict[image].strip().lower()

            refs.append([ref.split()])
            hyps.append(hyp.split())
            matched += 1

    print(f"共匹配图像数: {matched}")

    # === BLEU 分数 ===
    smooth = SmoothingFunction().method4
    print(f"BLEU-1: {corpus_bleu(refs, hyps, weights=(1, 0, 0, 0), smoothing_function=smooth):.4f}")
    print(f"BLEU-2: {corpus_bleu(refs, hyps, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth):.4f}")
    print(f"BLEU-3: {corpus_bleu(refs, hyps, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smooth):.4f}")
    print(f"BLEU-4: {corpus_bleu(refs, hyps, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth):.4f}")

    # === ROUGE-L ===
    rouge = Rouge()
    rouge_scores = rouge.get_scores([" ".join(h) for h in hyps], [" ".join(r[0]) for r in refs], avg=True)
    print(f"ROUGE-L F1: {rouge_scores['rouge-l']['f']:.4f}")
    print(f"ROUGE-L Precision: {rouge_scores['rouge-l']['p']:.4f}")
    print(f"ROUGE-L Recall: {rouge_scores['rouge-l']['r']:.4f}")

if __name__ == "__main__":
    evaluate(
        gold_file=r"E:\rubbercone-captioning\outputs\train_blip_natural.jsonl",   # 👈 替换为你数据路径
        pred_file=r"E:\rubbercone-captioning\pred_outputs\captions.jsonl" # 👈 替换为你推理结果路径
    )
