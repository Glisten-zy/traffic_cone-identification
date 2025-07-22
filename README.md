首先收集图片
然后进行标注
生成自然语言版本描述，作为blip训练集
训练模型
yolo预测训练
yolo输出格式转换成blip模型的输入
api集成
生成ui


改进：微调的时候加入描述位置的prompt

训练
cd yolov5
python train.py --img 640 --batch 16 --epochs 50 --data ../datasets/cone/cone.yaml --weights yolov5s.pt --name cone_detect

推理
cd yolov5

python detect.py --weights runs/train/cone_detect2/weights/best.pt --img 640 --source ../source_img --conf 0.25 --save-txt --save-conf --name cone_test


python run_captioning.py \
  --image_path E:/rubbercone-captioning/source_img/1906271639419637.jpg \
  --yolo_weights E:/rubbercone-captioning/yolov5/yolov5s.pt \
  --blip_dir E:/rubbercone-captioning/blip_finetuned/checkpoint-130 \
  --output_dir E:/rubbercone-captioning/outputs \
  --device cuda



uvicorn scripts.api_server:app --reload


python app.py


测试：
python yolov5/val.py --weights E:\rubbercone-captioning\yolov5\runs\train\cone_detect2\weights\best.pt --data E:\rubbercone-captioning\datasets\cone\cone.yaml --img 640 --task test --save-conf --save-json --save-txt --save-hybrid

输出预测：
python scripts/predict_pipeline.py --image_path E:/rubbercone-captioning/images/ --yolo_weights E:/rubbercone-captioning/yolov5/runs/train/cone_detect2/weights/best.pt --blip_dir E:/rubbercone-captioning/blip_finetuned/checkpoint-130 --output_dir E:/rubbercone-captioning/pred_outputs --device cpu

python scripts/eval.py --pred_file E:\rubbercone-captioning\pred_outputs\captions_cleaned.jsonl --ref_file E:\rubbercone-captioning\outputs\train_blip_natural.jsonl

