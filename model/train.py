import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from model.data_loading import dataloader
from tqdm import tqdm
from torchvision.models import resnet50
import pandas as pd
from torch.amp import autocast, GradScaler
import os
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet50(weights=None, num_classes=len(dataloader.dataset.label_idx)).to(device)

# 计算类别权重
train_df = pd.read_csv('/root/classify-leaves/train.csv')
label_counts = train_df['label'].value_counts().sort_index()
weights = len(train_df) / (len(label_counts) * label_counts.values)  # 归一化倒数法
class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.001)
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.4, patience=3, threshold=2e-3,min_lr=1e-6)
torch.backends.cudnn.benchmark = True  # 自动选择最优cuDNN算法

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# 加载断点
start_epoch = 0
if os.path.exists(os.path.join(CHECKPOINT_DIR, "latest.pth")):
    try:
        checkpoint = torch.load(os.path.join(CHECKPOINT_DIR, "latest.pth"), map_location=device)
        if "model_state" in checkpoint:
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            lr_scheduler.load_state_dict(checkpoint["scheduler_state"])
            start_epoch = checkpoint["epoch"] + 1
            print(f"Resumed from epoch {start_epoch}")
        else:
            # 兼容只保存了模型参数的情况
            model.load_state_dict(checkpoint)
            print("Loaded model weights only (no optimizer/scheduler state).")
            start_epoch = 0
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        start_epoch = 0

def train(model, dataloader, criterion, optimizer, num_epochs=10):
    model.train()
    scaler = GradScaler()  # 混合精度梯度缩放器
    global start_epoch
    for epoch in range(start_epoch, num_epochs):
        running_loss = 0.0
        for images, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            with autocast(device_type=device.type):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
        epoch_loss = running_loss / len(dataloader.dataset)
        lr_scheduler.step(epoch_loss)
        print(f"Loss: {epoch_loss:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        # 保存断点和模型
        checkpoint = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": lr_scheduler.state_dict(),
            "epoch": epoch
        }
        torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, "latest.pth"))
        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"model_epoch_{epoch}.pth"))
        # 保留最近5个模型
        model_files = sorted([f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("model_epoch_")],
                             key=lambda x: int(x.split("_")[-1].split(".")[0]))
        if len(model_files) > 5:
            for old_file in model_files[:-5]:
                os.remove(os.path.join(CHECKPOINT_DIR, old_file))

if __name__ == "__main__":
    train(model, dataloader, criterion, optimizer, num_epochs=150)
    print("Training complete and model saved.")
