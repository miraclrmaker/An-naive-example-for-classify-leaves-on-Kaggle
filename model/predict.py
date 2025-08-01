import torch
from torchvision.models import resnet50
from torchvision import transforms
from PIL import Image
import pandas as pd
from model.data_loading import dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
model = resnet50(weights=None, num_classes=176)
checkpoint = torch.load("checkpoints/latest.pth", map_location=device)
# 兼容两种保存方式：仅state_dict或包含'model_state'
if isinstance(checkpoint, dict) and "model_state" in checkpoint:
    model.load_state_dict(checkpoint["model_state"])
else:
    model.load_state_dict(checkpoint)
model = model.to(device)
model.eval()

# 反向标签映射
idx2label = {v: k for k, v in dataloader.dataset.label_idx.items()}

# 定义图片预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 读取测试集
test_df = pd.read_csv("/root/classify-leaves/test.csv")

results = []

with torch.no_grad():
    for i, row in test_df.iterrows():
        img_path = row['image']
        image = Image.open(img_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)
        output = model(image)
        pred_idx = output.argmax(dim=1).item()
        pred_label = idx2label[pred_idx]
        results.append({'image': img_path, 'label': pred_label})

# 保存预测结果
results_df = pd.DataFrame(results)
results_df.to_csv("test_predictions.csv", index=False)
print("预测完成，结果已保存到 test_predictions.csv")