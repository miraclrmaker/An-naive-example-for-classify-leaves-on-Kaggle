import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

csv = pd.read_csv("/root/classify-leaves/train.csv")

class LeafDataset(Dataset):
    def __init__(self,csv):
        self.csv = csv
        self.label_idx = {label : idx for idx, label in enumerate(csv['label'].unique())}
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    def __len__(self):
        return len(self.csv)
    def __getitem__(self, idx):
        img_path = self.csv.iloc[idx]['image']
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        label = self.label_idx[self.csv.iloc[idx]['label']]
        return image, label

dataset = LeafDataset(csv)
dataloader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    num_workers=8,           # 提高 worker 数量
    pin_memory=True,         # 加速 CPU 到 GPU 传输
    persistent_workers=True  # worker 持久化，减少重启开销
)
if __name__ == "__main__":
    print(len(dataset.label_idx))