# Vietnamese LPR Model Training Rules

## Overview

Training rules for YOLOv11 detection and PaddleOCR recognition models.

## YOLOv11 Detection Training

### Model Selection

| Model | Speed | Accuracy | Use Case |
|-------|-------|----------|----------|
| yolov11n | Fastest | Lowest | Real-time, edge devices |
| yolov11s | Fast | Good | **Recommended (balanced)** |
| yolov11m | Medium | Better | Higher accuracy needed |
| yolov11l | Slow | Best | Research, maximum accuracy |

### Training Configuration

```yaml
# configs/detection.yaml
model:
  name: yolov11s
  pretrained: true
  nc: 1  # plate only

training:
  epochs: 100
  batch_size: 16  # adjust for GPU memory
  imgsz: 640
  optimizer: AdamW
  lr0: 0.001
  warmup_epochs: 3
```

### Training Command

```bash
python -m src.detection.train \
    --data data/datasets/yolo_detection/data.yaml \
    --model yolov11s \
    --epochs 100 \
    --batch 16 \
    --device cuda
```

### Validation Metrics

| Metric | Target | Minimum |
|--------|--------|---------|
| mAP@0.5 | > 0.95 | 0.90 |
| mAP@0.5:0.95 | > 0.85 | 0.80 |
| Precision | > 0.95 | 0.90 |
| Recall | > 0.93 | 0.88 |

### Early Stopping
- Patience: 50 epochs
- Monitor: mAP@0.5
- Mode: max

## PaddleOCR Recognition Training

### Model Configuration

```yaml
# configs/recognition.yaml
Global:
  algorithm: CRNN
  max_text_length: 25
  character_dict_path: configs/vietnamese_dict.txt

Architecture:
  Backbone: ResNet34_vd
  Neck: RNNEncoder
  Head: CTCHead
```

### Data Preparation

1. Use trained detector to crop plates
2. Create label file: `image_name.jpg\tplate_text`
3. Minimum 5000 cropped plates for training

### Training Command

```bash
python -m src.recognition.train \
    --config configs/recognition.yaml \
    --epochs 200
```

### Validation Metrics

| Metric | Target | Minimum |
|--------|--------|---------|
| Character Accuracy | > 95% | 92% |
| Word Accuracy | > 90% | 85% |
| Mean Confidence | > 0.90 | 0.85 |

## Model Checkpoints

Save checkpoints at:
- `runs/train/{name}/weights/best.pt` - Best mAP
- `runs/train/{name}/weights/last.pt` - Last epoch

## Fine-tuning

### From Pretrained
```python
# Load pretrained and fine-tune
model = YOLO("yolov11s.pt")
model.train(
    data="data.yaml",
    epochs=50,  # fewer epochs for fine-tuning
    lr0=0.0001,  # lower LR
    freeze=10,  # freeze backbone
)
```

### Transfer Learning
- Freeze first 10 layers for 20 epochs
- Unfreeze and train with lower LR
- Use cosine scheduler

## Training Best Practices

1. **Start with pretrained weights** (never from scratch)
2. **Use validation set** for model selection
3. **Monitor overfitting** (train vs val loss gap)
4. **Data augmentation** is crucial for small datasets
5. **Batch size** affects convergence - use largest that fits GPU

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Low mAP | More data, augmentation, larger model |
| Overfitting | Regularization, dropout, more data |
| Slow training | Reduce batch size, mixed precision |
| Nan loss | Check data, reduce LR, check labels |
