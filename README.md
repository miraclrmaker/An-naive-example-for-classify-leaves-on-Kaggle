# Classify Leaves

This is a naive implementation for the [Kaggle Classify Leaves Competition](https://www.kaggle.com/competitions/classify-leaves/overview) (click here for more information).

> The model achieves **96% accuracy** on the test set after training for 150 epochs.


## Model Architecture

### Network
- **Model**: ResNet-50
- **Pretrained**: None (trained from scratch)
- **Output**: Fully connected layer for 176 classes
### Training Parameters
```python
- Input size: 224 × 224 × 3
- Optimizer: Adam (learning rate: 0.001)
- Loss function: CrossEntropyLoss (with class weights)
- Learning rate scheduler: ReduceLROnPlateau
  - Factor: 0.4
  - Patience: 3
  - Threshold: 2e-3
  - Min learning rate: 1e-6
```

## Data Preprocessing

### Training Data Augmentation
```python
transforms.Compose([
    transforms.Resize((224, 224)),           # Resize images
    transforms.RandomHorizontalFlip(),       # Random horizontal flip
    transforms.ToTensor(),                   # Convert to tensor
    transforms.Normalize(                    # ImageNet normalization
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])
```


## Training Configuration

### Class Weight Balancing
To address class imbalance issues, we use normalized inverse frequency weighting:
```python
weights = len(train_df) / (len(label_counts) * label_counts.values)
class_weights = torch.tensor(weights, dtype=torch.float32)
```

### Mixed Precision Training
Using PyTorch's AMP (Automatic Mixed Precision) for faster training:
- Automatic mixed precision (autocast)
- Gradient scaling (GradScaler)

### Checkpoint Management
- Automatic checkpoint saving after each epoch
- Resume training from interruption points
- Maintains the latest 5 model versions


## Requirements
```bash
pip install torch torchvision pandas tqdm pillow
```

