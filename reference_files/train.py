import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from torchvision import transforms, models, datasets
from PIL import Image, ImageFile

from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassF1Score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
import json
import pandas as pd
from datetime import datetime
from pathlib import Path

# Enable loading of truncated images (PIL will try to load what it can)
ImageFile.LOAD_TRUNCATED_IMAGES = True

#====================
# Robust Image Dataset Handler
#====================

class RobustImageFolder(Dataset):
    """
    A robust version of ImageFolder that handles corrupted/truncated images gracefully
    """
    def __init__(self, root, transform=None, extensions=('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
        self.root = root
        self.transform = transform
        self.extensions = extensions
        
        # Find all classes and create class_to_idx mapping
        self.classes = sorted([d.name for d in Path(root).iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        # Collect all image paths without validation
        self.samples = []
        self._collect_image_paths()
        
        print(f"Found {len(self.samples)} image files in {root}")
    
    def _collect_image_paths(self):
        """Collect all image file paths without validation"""
        for class_name in self.classes:
            class_dir = Path(self.root) / class_name
            class_idx = self.class_to_idx[class_name]
            
            for img_path in class_dir.rglob('*'):
                if img_path.suffix.lower() in self.extensions:
                    self.samples.append((str(img_path), class_idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get item with robust error handling"""
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                img_path, class_idx = self.samples[idx]
                
                # Load image with error handling
                image = self._load_image_safely(img_path)
                if image is None:
                    # If current image fails, try a random valid one
                    idx = torch.randint(0, len(self.samples), (1,)).item()
                    continue
                
                # Apply transforms
                if self.transform:
                    image = self.transform(image)
                
                return image, class_idx
                
            except Exception as e:
                print(f"Skipping corrupted image at index {idx}: {e}")
                # Try a different random index
                idx = torch.randint(0, len(self.samples), (1,)).item()
                if attempt == max_attempts - 1:
                    # Last attempt failed, return a dummy image
                    return self._get_dummy_image(), 0
        
        return self._get_dummy_image(), 0
    
    def _load_image_safely(self, img_path):
        """Safely load an image file"""
        try:
            with Image.open(img_path) as img:
                # Convert to RGB (handles grayscale, RGBA, palette images)
                image = img.convert('RGB')
                # Verify the image was loaded correctly
                if image.width < 10 or image.height < 10:
                    return None
                return image
        except Exception as e:
            print(f"Corrupted image: {img_path} - {e}")
            return None
    
    def _get_dummy_image(self):
        """Create a dummy image as fallback"""
        if self.transform:
            # Create a dummy image that matches expected input size
            dummy = Image.new('RGB', (224, 224), color='gray')
            return self.transform(dummy)
        else:
            return torch.zeros(3, 224, 224)

def robust_collate_fn(batch):
    """Custom collate function that filters out None samples"""
    # Filter out None samples
    batch = [item for item in batch if item is not None and item[0] is not None]
    
    if len(batch) == 0:
        # If all samples are bad, return dummy batch
        dummy_image = torch.zeros(1, 3, 224, 224)
        dummy_label = torch.zeros(1, dtype=torch.long)
        return dummy_image, dummy_label
    
    # Use default collate for valid samples
    return torch.utils.data.default_collate(batch)

#====================
# 0. Create directories for saving outputs
#====================
os.makedirs('./results', exist_ok=True)
os.makedirs('./results/plots', exist_ok=True)
os.makedirs('./results/logs', exist_ok=True)
os.makedirs('./models', exist_ok=True)

# Create timestamp for this training run
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

#====================
# EfficientNet Model Selection
#====================
print("Available EfficientNet Models:")
print("="*60)
print("B0: Baseline model (5.3M params, 224x224) - Fastest, lowest accuracy")
print("B1: (7.8M params, 240x240) - Good balance")
print("B2: (9.2M params, 260x260) - Better accuracy")
print("B3: (12M params, 300x300) - Higher accuracy")
print("B4: (19M params, 380x380) - Very high accuracy")
print("B5: (30M params, 456x456) - Excellent accuracy")
print("B6: (43M params, 528x528) - Top accuracy")
print("B7: (66M params, 600x600) - Highest accuracy, slowest")
print("="*60)

# Model selection - Change this to choose your preferred model
EFFICIENTNET_VERSION = "B0"  # Options: B0, B1, B2, B3, B4, B5, B6, B7

# Get model configuration
def get_efficientnet_config(version):
    configs = {
        "B0": {
            "model_fn": models.efficientnet_b0,
            "weights": models.EfficientNet_B0_Weights.IMAGENET1K_V1,
            "input_size": 224,
            "params": "5.3M",
            "description": "Baseline - Fast training, good for quick experiments"
        },
        "B1": {
            "model_fn": models.efficientnet_b1,
            "weights": models.EfficientNet_B1_Weights.IMAGENET1K_V2,
            "input_size": 240,
            "params": "7.8M",
            "description": "Good balance of speed and accuracy"
        },
        "B2": {
            "model_fn": models.efficientnet_b2,
            "weights": models.EfficientNet_B2_Weights.IMAGENET1K_V1,
            "input_size": 260,
            "params": "9.2M",
            "description": "Better accuracy with moderate increase in size"
        },
        "B3": {
            "model_fn": models.efficientnet_b3,
            "weights": models.EfficientNet_B3_Weights.IMAGENET1K_V1,
            "input_size": 300,
            "params": "12M",
            "description": "Higher accuracy, still reasonable training time"
        },
        "B4": {
            "model_fn": models.efficientnet_b4,
            "weights": models.EfficientNet_B4_Weights.IMAGENET1K_V1,
            "input_size": 380,
            "params": "19M",
            "description": "Very high accuracy, longer training time"
        },
        "B5": {
            "model_fn": models.efficientnet_b5,
            "weights": models.EfficientNet_B5_Weights.IMAGENET1K_V1,
            "input_size": 456,
            "params": "30M",
            "description": "Excellent accuracy, requires more memory"
        },
        "B6": {
            "model_fn": models.efficientnet_b6,
            "weights": models.EfficientNet_B6_Weights.IMAGENET1K_V1,
            "input_size": 528,
            "params": "43M",
            "description": "Top-tier accuracy, high memory requirements"
        },
        "B7": {
            "model_fn": models.efficientnet_b7,
            "weights": models.EfficientNet_B7_Weights.IMAGENET1K_V1,
            "input_size": 600,
            "params": "66M",
            "description": "Highest accuracy, longest training time"
        }
    }
    
    if version not in configs:
        print(f"Invalid EfficientNet version: {version}")
        print(f"Available versions: {list(configs.keys())}")
        exit(1)
    
    return configs[version]

# Get selected model configuration
model_config = get_efficientnet_config(EFFICIENTNET_VERSION)
INPUT_SIZE = model_config["input_size"]

print(f"\nSelected Model: EfficientNet-{EFFICIENTNET_VERSION}")
print(f"Input Size: {INPUT_SIZE}x{INPUT_SIZE}")
print(f"Parameters: {model_config['params']}")
print(f"Description: {model_config['description']}")
print("="*60)

# Update run name with selected model
run_name = f"efficientnet_{EFFICIENTNET_VERSION.lower()}_{timestamp}"

#====================
# 1. Configuration
#====================
CLASSES        = ['Drawing', 'Hentai', 'Neutral', 'Porn', 'Sexy']
NUM_CLASSES    = len(CLASSES)
TRAIN_DIR      = '../../LSPD/train/'
VALID_DIR      = '../../LSPD/val/'
TEST_DIR       = '../../LSPD/test/'

BATCH_SIZE     = 32
NUM_WORKERS    = min(16, os.cpu_count())  # Use more CPU cores for data loading
PREFETCH_FACTOR = 4  # Prefetch 4 batches per worker
DEVICE         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_EPOCHS     = 30
PATIENCE       = 5
LR_BACKBONE    = 1e-5
LR_HEAD        = 1e-4
WEIGHT_DECAY   = 1e-4
FREEZE_EPOCHS  = 5  # train head only for this many epochs

# Adjust batch size for larger models to prevent OOM
if EFFICIENTNET_VERSION in ["B6", "B7"]:
    BATCH_SIZE = 16
    print(f"Reduced batch size to {BATCH_SIZE} for {EFFICIENTNET_VERSION} to prevent OOM")
elif EFFICIENTNET_VERSION in ["B4", "B5"]:
    BATCH_SIZE = 24
    print(f"Reduced batch size to {BATCH_SIZE} for {EFFICIENTNET_VERSION}")

# Data loading optimization settings
PERSISTENT_WORKERS = True  # Keep workers alive between epochs
PIN_MEMORY = True if torch.cuda.is_available() else False

print(f"Using {NUM_WORKERS} workers for data loading")
print(f"Prefetch factor: {PREFETCH_FACTOR}")
print(f"Device: {DEVICE}")
print(f"Batch size: {BATCH_SIZE}")

#====================
# Initialize logging
#====================
training_log = []
config_info = {
    'run_name': run_name,
    'timestamp': timestamp,
    'model': f'EfficientNet-{EFFICIENTNET_VERSION}',
    'model_params': model_config['params'],
    'classes': CLASSES,
    'num_classes': NUM_CLASSES,
    'input_size': INPUT_SIZE,
    'batch_size': BATCH_SIZE,
    'num_epochs': NUM_EPOCHS,
    'lr_backbone': LR_BACKBONE,
    'lr_head': LR_HEAD,
    'weight_decay': WEIGHT_DECAY,
    'freeze_epochs': FREEZE_EPOCHS,
    'device': str(DEVICE)
}

#====================
# 2. Optimized Transforms with model-specific input size
#====================
# Optimized training transforms - efficient for CPU processing
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(INPUT_SIZE, scale=(0.8, 1.0), antialias=True),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.RandomRotation(10),  # Additional augmentation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Validation transforms - fast and minimal
val_transforms = transforms.Compose([
    transforms.Resize(INPUT_SIZE + 32, antialias=True),
    transforms.CenterCrop(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

#====================
# 3. Robust Datasets & Loaders
#====================
print("Creating robust datasets...")

# Create datasets using robust image folder
train_dataset = RobustImageFolder(TRAIN_DIR, transform=train_transforms)
val_dataset = RobustImageFolder(VALID_DIR, transform=val_transforms)
test_dataset = RobustImageFolder(TEST_DIR, transform=val_transforms)

# Compute class counts & weights
print("Computing class distribution and weights...")
class_counts = [0] * NUM_CLASSES
for _, label in train_dataset.samples:
    class_counts[label] += 1

print(f"Class distribution in training set:")
total_samples = sum(class_counts)
class_distribution = {}
for i, (class_name, count) in enumerate(zip(CLASSES, class_counts)):
    percentage = (count / total_samples) * 100
    print(f"  {class_name}: {count} samples ({percentage:.1f}%)")
    class_distribution[class_name] = {'count': count, 'percentage': percentage}

# Method 1: Inverse frequency weighting
class_weights_inv = [total_samples / (NUM_CLASSES * count) for count in class_counts]

# Method 2: Balanced class weighting (sklearn style)
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
labels_array = np.array([label for _, label in train_dataset.samples])
class_weights_balanced = compute_class_weight('balanced', classes=np.unique(labels_array), y=labels_array)

print(f"\nClass weights (inverse frequency): {[f'{w:.3f}' for w in class_weights_inv]}")
print(f"Class weights (balanced): {[f'{w:.3f}' for w in class_weights_balanced]}")

# Use balanced weights (generally better performance)
class_weights = class_weights_balanced.tolist()

# Save class distribution and weights
config_info['class_distribution'] = class_distribution
config_info['class_weights'] = {name: weight for name, weight in zip(CLASSES, class_weights)}

# Create sample weights for WeightedRandomSampler
sample_weights = [class_weights[label] for _, label in train_dataset.samples]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_dataset), replacement=True)

print(f"Created WeightedRandomSampler with {len(sample_weights)} sample weights")

# Create robust dataloaders with custom collate function
train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    sampler=sampler,
    num_workers=NUM_WORKERS, 
    pin_memory=PIN_MEMORY,
    prefetch_factor=PREFETCH_FACTOR,
    persistent_workers=PERSISTENT_WORKERS,
    drop_last=True,  # Ensures consistent batch sizes
    collate_fn=robust_collate_fn  # Use custom collate function
)
val_loader = DataLoader(
    val_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False,
    num_workers=NUM_WORKERS, 
    pin_memory=PIN_MEMORY,
    prefetch_factor=PREFETCH_FACTOR,
    persistent_workers=PERSISTENT_WORKERS,
    drop_last=False,
    collate_fn=robust_collate_fn
)
test_loader = DataLoader(
    test_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False,
    num_workers=NUM_WORKERS, 
    pin_memory=PIN_MEMORY,
    prefetch_factor=PREFETCH_FACTOR,
    persistent_workers=PERSISTENT_WORKERS,
    drop_last=False,
    collate_fn=robust_collate_fn
)

#====================
# 4. EfficientNet Model Definition
#====================
class NSFWClassifier(nn.Module):
    def __init__(self, efficientnet_version=EFFICIENTNET_VERSION, num_classes=NUM_CLASSES, pretrained=True):
        super().__init__()
        
        # Get the appropriate EfficientNet model
        config = get_efficientnet_config(efficientnet_version)
        
        if pretrained:
            backbone = config["model_fn"](weights=config["weights"])
        else:
            backbone = config["model_fn"](weights=None)
            
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        
        # Get the number of features from the backbone
        in_feats = backbone.classifier[1].in_features
        
        # Create custom classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2, inplace=True),
            nn.Linear(in_feats, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        self.efficientnet_version = efficientnet_version
        self._freeze_backbone()

    def _freeze_backbone(self):
        for p in self.features.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self, layers=50):
        children = list(self.features.children())
        # For EfficientNet, unfreeze the last few stages
        num_to_unfreeze = min(layers, len(children))
        for layer in children[-num_to_unfreeze:]:
            for p in layer.parameters():
                p.requires_grad = True

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

model = NSFWClassifier().to(DEVICE)

#====================
# Print Model Information
#====================
def print_model_info():
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Information:")
    print(f"="*60)
    print(f"Architecture: EfficientNet-{EFFICIENTNET_VERSION}")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Frozen Parameters: {total_params - trainable_params:,}")
    print(f"Model Size: {total_params * 4 / (1024 * 1024):.2f} MB")
    print(f"Input Size: {INPUT_SIZE}x{INPUT_SIZE}")
    print(f"="*60)

print_model_info()

#====================
# 5. Optimizer & Scheduler Setup Function
#====================
def make_optimizer():
    params = []
    # always train head
    params.append({'params': model.classifier.parameters(), 'lr': LR_HEAD})
    # after unfreeze, backbone params have requires_grad=True
    backbone_params = [p for p in model.features.parameters() if p.requires_grad]
    if backbone_params:
        params.append({'params': backbone_params, 'lr': LR_BACKBONE})
    return optim.AdamW(params, weight_decay=WEIGHT_DECAY)

def make_scheduler(optimizer, total_steps):
    # Build max_lr matching groups: single value or list
    group_lrs = [group['lr'] for group in optimizer.param_groups]
    max_lr = group_lrs[0] if len(group_lrs) == 1 else group_lrs
    return optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total_steps
    )

# Initial optimizer & scheduler for head-only
torch.manual_seed(42)
optimizer = make_optimizer()
total_steps = NUM_EPOCHS * len(train_loader)
scheduler = make_scheduler(optimizer, total_steps)
criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, device=DEVICE))

#====================
# 6. Metrics & Storage
#====================
precision = MulticlassPrecision(num_classes=NUM_CLASSES, average='none').to(DEVICE)
recall    = MulticlassRecall(num_classes=NUM_CLASSES, average='none').to(DEVICE)
f1_macro  = MulticlassF1Score(num_classes=NUM_CLASSES, average='macro').to(DEVICE)

history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'val_f1': []}
best_f1 = 0.0
epochs_no_improve = 0

#====================
# 7. Robust Train & Eval Functions with Error Handling
#====================
def robust_train_one_epoch(epoch):
    """Training loop with robust error handling"""
    model.train()
    running_loss, correct, total = 0, 0, 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]", leave=False)
    
    for batch_idx, (images, labels) in enumerate(loop):
        try:
            # Skip empty batches
            if images.size(0) == 0:
                continue
                
            # Async data transfer to GPU with non_blocking=True
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Statistics (move to CPU only once)
            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            # Update progress bar every few batches to reduce overhead
            if batch_idx % 10 == 0:
                loop.set_postfix(
                    loss=f"{running_loss/max(total, 1):.4f}", 
                    acc=f"{correct/max(total, 1):.4f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}"
                )
                
        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            continue  # Skip this batch and continue training
    
    if total == 0:
        return 0.0, 0.0
    
    history['train_loss'].append(running_loss/total)
    history['train_acc'].append(correct/total)

def robust_validate_one_epoch(epoch):
    """Validation loop with robust error handling"""
    model.eval()
    running_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]", leave=False)
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loop):
            try:
                # Skip empty batches
                if images.size(0) == 0:
                    continue
                    
                # Async data transfer
                images = images.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)
                
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                # Statistics
                running_loss += loss.item() * images.size(0)
                preds = outputs.argmax(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
                # Collect predictions (keep on GPU until end)
                all_preds.append(preds)
                all_labels.append(labels)
                
                # Update progress bar every few batches
                if batch_idx % 10 == 0:
                    loop.set_postfix(
                        loss=f"{running_loss/max(total, 1):.4f}", 
                        acc=f"{correct/max(total, 1):.4f}"
                    )
                    
            except Exception as e:
                print(f"Error in validation batch {batch_idx}: {e}")
                continue  # Skip this batch and continue
    
    if total == 0:
        return 0.0, 0.0
    
    history['val_loss'].append(running_loss/total)
    history['val_acc'].append(correct/total)
    
    # Compute F1 score (single GPU operation)
    if all_preds and all_labels:
        preds = torch.cat(all_preds)
        labels = torch.cat(all_labels)
        f1 = f1_macro(preds, labels).item()
        history['val_f1'].append(f1)
    else:
        history['val_f1'].append(0.0)
        f1 = 0.0
    
    return history['val_acc'][-1], f1

#====================
# 8. Main Training Loop with Unfreeze
#====================
print(f"\nStarting training - Run: {run_name}")
print("="*60)

for epoch in range(NUM_EPOCHS):
    epoch_start_time = datetime.now()
    
    # Unfreeze backbone after FREEZE_EPOCHS
    if epoch == FREEZE_EPOCHS:
        model.unfreeze_backbone()
        optimizer = make_optimizer()
        scheduler = make_scheduler(optimizer, (NUM_EPOCHS-epoch)*len(train_loader))
        print(f"Backbone unfrozen at epoch {epoch+1}")

    robust_train_one_epoch(epoch)
    val_acc, val_f1 = robust_validate_one_epoch(epoch)
    
    epoch_time = (datetime.now() - epoch_start_time).total_seconds()
    
    # Log epoch results
    epoch_log = {
        'epoch': epoch + 1,
        'train_acc': history['train_acc'][-1],
        'train_loss': history['train_loss'][-1],
        'val_acc': val_acc,
        'val_loss': history['val_loss'][-1],
        'val_f1': val_f1,
        'epoch_time_seconds': epoch_time,
        'learning_rate': scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else 'N/A',
        'backbone_frozen': epoch < FREEZE_EPOCHS
    }
    training_log.append(epoch_log)
    
    # Print epoch summary
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Acc: {history['train_acc'][-1]:.4f}, "
          f"Val Acc: {val_acc:.4f} | Train Loss: {history['train_loss'][-1]:.4f}, "
          f"Val Loss: {history['val_loss'][-1]:.4f} | Val F1: {val_f1:.4f} | Time: {epoch_time:.1f}s")

    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(model.state_dict(), f'./models/{run_name}_best.pth')
        print(f"  → New best model saved! F1: {best_f1:.4f}")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

print(f"\nTraining completed! Best validation Macro-F1: {best_f1:.4f}")

# Save training logs
with open(f'./results/logs/{run_name}_training_log.json', 'w') as f:
    json.dump({
        'config': config_info,
        'training_log': training_log,
        'best_f1': best_f1,
        'total_epochs': len(training_log)
    }, f, indent=2)

# Save training log as CSV for easy analysis
training_df = pd.DataFrame(training_log)
training_df.to_csv(f'./results/logs/{run_name}_training_log.csv', index=False)

#====================
# 9. Plot and Save Training History
#====================
def save_training_plots():
    plt.style.use('default')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    epochs = range(1, len(history['train_acc']) + 1)
    
    # Accuracy plot
    ax1.plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    ax1.plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    ax1.set_title(f'Training and Validation Accuracy - EfficientNet-{EFFICIENTNET_VERSION}', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # Loss plot
    ax2.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax2.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax2.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # F1 Score plot
    ax3.plot(epochs, history['val_f1'], 'g-', label='Val F1-Score', linewidth=2)
    ax3.set_title('Validation F1-Score', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('F1-Score')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)
    
    # Learning rate plot
    lrs = [log['learning_rate'] for log in training_log if log['learning_rate'] != 'N/A']
    if lrs:
        ax4.plot(epochs[:len(lrs)], lrs, 'purple', label='Learning Rate', linewidth=2)
        ax4.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Learning Rate')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(f'./results/plots/{run_name}_training_history.png', dpi=300, bbox_inches='tight')

save_training_plots()

#====================
# 10. Robust Test Evaluation
#====================
print("\n" + "="*60)
print("STARTING TEST EVALUATION")
print("="*60)

print("Loading best model and running test evaluation...")
model.load_state_dict(torch.load(f'./models/{run_name}_best.pth', map_location=DEVICE))
model.eval()

all_preds, all_labels = [], []
test_loop = tqdm(test_loader, desc="Testing", leave=True)

with torch.no_grad():
    for images, labels in test_loop:
        try:
            # Skip empty batches
            if images.size(0) == 0:
                continue
                
            # Async data transfer
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            
            # Inference
            outputs = model(images)
            preds = outputs.argmax(1)
            
            # Collect results (move to CPU at the end)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            
        except Exception as e:
            print(f"Error in test batch: {e}")
            continue

# Convert to numpy for sklearn metrics
if all_preds and all_labels:
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
else:
    print("Warning: No valid test samples found!")
    preds = np.array([])
    labels = np.array([])

# Calculate test metrics
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

if len(preds) > 0 and len(labels) > 0:
    test_accuracy = accuracy_score(labels, preds)
    test_f1_macro = f1_score(labels, preds, average='macro')
    test_f1_weighted = f1_score(labels, preds, average='weighted')
    test_precision = precision_score(labels, preds, average='macro')
    test_recall = recall_score(labels, preds, average='macro')
else:
    test_accuracy = 0.0
    test_f1_macro = 0.0
    test_f1_weighted = 0.0
    test_precision = 0.0
    test_recall = 0.0

# Save test results
test_results = {
    'test_accuracy': test_accuracy,
    'test_f1_macro': test_f1_macro,
    'test_f1_weighted': test_f1_weighted,
    'test_precision': test_precision,
    'test_recall': test_recall,
    'best_val_f1': best_f1
}

#====================
# 11. Generate and Save Confusion Matrix
#====================
def save_confusion_matrix():
    if len(preds) == 0 or len(labels) == 0:
        print("Warning: Cannot create confusion matrix - no valid predictions")
        return np.zeros((NUM_CLASSES, NUM_CLASSES))
        
    cm = confusion_matrix(labels, preds)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=CLASSES, yticklabels=CLASSES, 
                cmap='Blues', cbar_kws={'shrink': 0.8}, square=True)
    plt.title(f'Confusion Matrix - EfficientNet-{EFFICIENTNET_VERSION}\nAccuracy: {test_accuracy:.4f} | F1-Score: {test_f1_macro:.4f}', 
              fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    
    # Add percentage annotations
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            plt.text(j+0.5, i+0.7, f'({cm_percent[i,j]:.1f}%)', 
                    ha='center', va='center', fontsize=10, color='gray')
    
    plt.tight_layout()
    plt.savefig(f'./results/plots/{run_name}_confusion_matrix.png', dpi=300, bbox_inches='tight')
    
    return cm

cm = save_confusion_matrix()

#====================
# 12. Generate and Save Classification Report
#====================
def save_classification_report():
    if len(preds) == 0 or len(labels) == 0:
        print("Warning: Cannot create classification report - no valid predictions")
        return {}
        
    report = classification_report(labels, preds, target_names=CLASSES, digits=4, output_dict=True)
    
    # Print to console
    print("\nTest Set Classification Report:")
    print("="*60)
    print(classification_report(labels, preds, target_names=CLASSES, digits=4))
    
    # Save as JSON
    with open(f'./results/logs/{run_name}_classification_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    # Create and save detailed classification report as CSV
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(f'./results/logs/{run_name}_classification_report.csv')
    
    # Create a visual classification report
    plt.figure(figsize=(12, 8))
    
    # Extract metrics for plotting
    classes_metrics = []
    for class_name in CLASSES:
        if class_name in report:
            classes_metrics.append([
                report[class_name]['precision'],
                report[class_name]['recall'],
                report[class_name]['f1-score']
            ])
        else:
            classes_metrics.append([0.0, 0.0, 0.0])
    
    classes_metrics = np.array(classes_metrics)
    
    # Create bar plot
    x = np.arange(len(CLASSES))
    width = 0.25
    
    plt.bar(x - width, classes_metrics[:, 0], width, label='Precision', alpha=0.8)
    plt.bar(x, classes_metrics[:, 1], width, label='Recall', alpha=0.8)
    plt.bar(x + width, classes_metrics[:, 2], width, label='F1-Score', alpha=0.8)
    
    plt.xlabel('Classes', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title(f'Per-Class Performance Metrics - EfficientNet-{EFFICIENTNET_VERSION}\nOverall Accuracy: {test_accuracy:.4f}', fontsize=14, fontweight='bold')
    plt.xticks(x, CLASSES, rotation=45)
    plt.legend()
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'./results/plots/{run_name}_per_class_metrics.png', dpi=300, bbox_inches='tight')
    
    return report

report = save_classification_report()

#====================
# 13. Save Final Results Summary
#====================
def save_final_summary():
    final_summary = {
        **config_info,
        **test_results,
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'training_summary': {
            'total_epochs_trained': len(training_log),
            'early_stopped': len(training_log) < NUM_EPOCHS,
            'best_epoch': max(training_log, key=lambda x: x['val_f1'])['epoch'] if training_log else 0,
            'final_train_acc': history['train_acc'][-1] if history['train_acc'] else 0,
            'final_val_acc': history['val_acc'][-1] if history['val_acc'] else 0,
            'final_val_f1': history['val_f1'][-1] if history['val_f1'] else 0
        }
    }
    
    # Save complete summary
    with open(f'./results/logs/{run_name}_final_summary.json', 'w') as f:
        json.dump(final_summary, f, indent=2)
    
    # Create summary text file
    with open(f'./results/logs/{run_name}_summary.txt', 'w') as f:
        f.write(f"NSFW Classification Training Summary\n")
        f.write(f"Run: {run_name}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"{'='*60}\n\n")
        
        f.write(f"CONFIGURATION:\n")
        f.write(f"- Model: EfficientNet-{EFFICIENTNET_VERSION}\n")
        f.write(f"- Model Parameters: {model_config['params']}\n")
        f.write(f"- Classes: {CLASSES}\n")
        f.write(f"- Input Size: {INPUT_SIZE}x{INPUT_SIZE}\n")
        f.write(f"- Batch Size: {BATCH_SIZE}\n")
        f.write(f"- Max Epochs: {NUM_EPOCHS}\n")
        f.write(f"- Device: {DEVICE}\n\n")
        
        f.write(f"TRAINING RESULTS:\n")
        f.write(f"- Total Epochs Trained: {len(training_log)}\n")
        f.write(f"- Best Validation F1: {best_f1:.4f}\n")
        f.write(f"- Final Train Accuracy: {history['train_acc'][-1] if history['train_acc'] else 0:.4f}\n")
        f.write(f"- Final Validation Accuracy: {history['val_acc'][-1] if history['val_acc'] else 0:.4f}\n\n")
        
        f.write(f"TEST RESULTS:\n")
        f.write(f"- Test Accuracy: {test_accuracy:.4f}\n")
        f.write(f"- Test F1-Score (Macro): {test_f1_macro:.4f}\n")
        f.write(f"- Test F1-Score (Weighted): {test_f1_weighted:.4f}\n")
        f.write(f"- Test Precision (Macro): {test_precision:.4f}\n")
        f.write(f"- Test Recall (Macro): {test_recall:.4f}\n\n")
        
        f.write(f"FILES SAVED:\n")
        f.write(f"- Model: ./models/{run_name}_best.pth\n")
        f.write(f"- Training History Plot: ./results/plots/{run_name}_training_history.png\n")
        f.write(f"- Confusion Matrix: ./results/plots/{run_name}_confusion_matrix.png\n")
        f.write(f"- Per-Class Metrics: ./results/plots/{run_name}_per_class_metrics.png\n")
        f.write(f"- Training Log (JSON): ./results/logs/{run_name}_training_log.json\n")
        f.write(f"- Training Log (CSV): ./results/logs/{run_name}_training_log.csv\n")
        f.write(f"- Classification Report: ./results/logs/{run_name}_classification_report.json\n")

save_final_summary()

print(f"\nTest Set Summary:")
print(f"Accuracy: {test_accuracy:.4f}")
print(f"F1-Score (Macro): {test_f1_macro:.4f}")
print(f"F1-Score (Weighted): {test_f1_weighted:.4f}")

print(f"\n{'='*60}")
print(f"TRAINING COMPLETED SUCCESSFULLY!")
print(f"Run: {run_name}")
print(f"Model: EfficientNet-{EFFICIENTNET_VERSION}")
print(f"Best Validation F1: {best_f1:.4f}")
print(f"Test F1-Score: {test_f1_macro:.4f}")
print(f"All results saved to ./results/")
print(f"{'='*60}")

#====================
# 14. Additional Analysis and Plots
#====================

def create_epoch_progression_plot():
    """Create detailed epoch-by-epoch progression plots"""
    if len(training_log) == 0:
        print("No training log available for plotting")
        return
        
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Detailed Training Analysis - EfficientNet-{EFFICIENTNET_VERSION}', fontsize=16, fontweight='bold')
    
    epochs = range(1, len(training_log) + 1)
    
    # Extract data from training log
    train_accs = [log['train_acc'] for log in training_log]
    val_accs = [log['val_acc'] for log in training_log]
    train_losses = [log['train_loss'] for log in training_log]
    val_losses = [log['val_loss'] for log in training_log]
    val_f1s = [log['val_f1'] for log in training_log]
    epoch_times = [log['epoch_time_seconds'] for log in training_log]
    
    # Accuracy comparison
    axes[0, 0].plot(epochs, train_accs, 'b-o', label='Train Acc', markersize=4, linewidth=2)
    axes[0, 0].plot(epochs, val_accs, 'r-s', label='Val Acc', markersize=4, linewidth=2)
    axes[0, 0].set_title('Accuracy Progression', fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(0, 1)
    
    # Loss comparison
    axes[0, 1].plot(epochs, train_losses, 'b-o', label='Train Loss', markersize=4, linewidth=2)
    axes[0, 1].plot(epochs, val_losses, 'r-s', label='Val Loss', markersize=4, linewidth=2)
    axes[0, 1].set_title('Loss Progression', fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # F1 Score progression
    axes[0, 2].plot(epochs, val_f1s, 'g-^', label='Val F1', markersize=4, linewidth=2)
    axes[0, 2].axhline(y=best_f1, color='orange', linestyle='--', label=f'Best F1 ({best_f1:.3f})')
    axes[0, 2].set_title('F1-Score Progression', fontweight='bold')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('F1-Score')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_ylim(0, 1)
    
    # Training time per epoch
    axes[1, 0].bar(epochs, epoch_times, alpha=0.7, color='purple')
    axes[1, 0].set_title('Training Time per Epoch', fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Time (seconds)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Gap between train and val accuracy
    acc_gap = [train_acc - val_acc for train_acc, val_acc in zip(train_accs, val_accs)]
    axes[1, 1].plot(epochs, acc_gap, 'orange', marker='o', markersize=4, linewidth=2)
    axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[1, 1].set_title('Train-Val Accuracy Gap', fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy Gap')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Cumulative training time
    cumulative_time = np.cumsum(epoch_times)
    axes[1, 2].plot(epochs, cumulative_time/60, 'brown', marker='s', markersize=4, linewidth=2)
    axes[1, 2].set_title('Cumulative Training Time', fontweight='bold')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Time (minutes)')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'./results/plots/{run_name}_detailed_analysis.png', dpi=300, bbox_inches='tight')

create_epoch_progression_plot()

def create_class_distribution_plot():
    """Create a visualization of class distribution"""
    plt.figure(figsize=(12, 8))
    
    class_names = list(class_distribution.keys())
    counts = [class_distribution[name]['count'] for name in class_names]
    percentages = [class_distribution[name]['percentage'] for name in class_names]
    
    # Create subplot for count and percentage
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bar plot for counts
    bars1 = ax1.bar(class_names, counts, alpha=0.8, color=plt.cm.Set3(np.linspace(0, 1, len(class_names))))
    ax1.set_title('Class Distribution - Sample Counts', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Number of Samples')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add count labels on bars
    for bar, count in zip(bars1, counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    # Pie chart for percentages
    colors = plt.cm.Set3(np.linspace(0, 1, len(class_names)))
    wedges, texts, autotexts = ax2.pie(percentages, labels=class_names, autopct='%1.1f%%',
                                       colors=colors, startangle=90)
    ax2.set_title('Class Distribution - Percentages', fontsize=14, fontweight='bold')
    
    # Make percentage text bold
    for autotext in autotexts:
        autotext.set_fontweight('bold')
    
    plt.tight_layout()
    plt.savefig(f'./results/plots/{run_name}_class_distribution.png', dpi=300, bbox_inches='tight')

create_class_distribution_plot()

def create_efficientnet_comparison_info():
    """Create a comparison chart of EfficientNet versions"""
    
    # EfficientNet comparison data
    efficientnet_versions = ['B0', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7']
    params_millions = [5.3, 7.8, 9.2, 12, 19, 30, 43, 66]
    input_sizes = [224, 240, 260, 300, 380, 456, 528, 600]
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Parameters comparison
    bars1 = ax1.bar(efficientnet_versions, params_millions, alpha=0.8, 
                    color=['red' if v == EFFICIENTNET_VERSION else 'lightblue' for v in efficientnet_versions])
    ax1.set_title('EfficientNet Parameters Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Parameters (Millions)')
    ax1.set_xlabel('EfficientNet Version')
    
    # Highlight selected model
    for i, (ver, params) in enumerate(zip(efficientnet_versions, params_millions)):
        color = 'white' if ver == EFFICIENTNET_VERSION else 'black'
        weight = 'bold' if ver == EFFICIENTNET_VERSION else 'normal'
        ax1.text(i, params + 1, f'{params}M', ha='center', va='bottom', 
                color=color, fontweight=weight)
    
    # Input size comparison
    bars2 = ax2.bar(efficientnet_versions, input_sizes, alpha=0.8,
                    color=['red' if v == EFFICIENTNET_VERSION else 'lightgreen' for v in efficientnet_versions])
    ax2.set_title('EfficientNet Input Size Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Input Size (pixels)')
    ax2.set_xlabel('EfficientNet Version')
    
    # Highlight selected model
    for i, (ver, size) in enumerate(zip(efficientnet_versions, input_sizes)):
        color = 'white' if ver == EFFICIENTNET_VERSION else 'black'
        weight = 'bold' if ver == EFFICIENTNET_VERSION else 'normal'
        ax2.text(i, size + 10, f'{size}px', ha='center', va='bottom',
                color=color, fontweight=weight)
    
    plt.tight_layout()
    plt.savefig(f'./results/plots/{run_name}_efficientnet_comparison.png', dpi=300, bbox_inches='tight')
    
    # Save comparison data
    comparison_data = {
        'selected_model': EFFICIENTNET_VERSION,
        'selected_params': model_config['params'],
        'selected_input_size': INPUT_SIZE,
        'all_versions': {
            ver: {
                'params_millions': params,
                'input_size': size,
                'selected': ver == EFFICIENTNET_VERSION
            }
            for ver, params, size in zip(efficientnet_versions, params_millions, input_sizes)
        }
    }
    
    with open(f'./results/logs/{run_name}_efficientnet_comparison.json', 'w') as f:
        json.dump(comparison_data, f, indent=2)

create_efficientnet_comparison_info()

print(f"\n{'='*60}")
print(f"ADDITIONAL ANALYSIS COMPLETED!")
print(f"{'='*60}")
print(f"All files saved in ./results/:")
print(f"Plots:")
print(f"  - {run_name}_training_history.png")
print(f"  - {run_name}_confusion_matrix.png") 
print(f"  - {run_name}_per_class_metrics.png")
print(f"  - {run_name}_detailed_analysis.png")
print(f"  - {run_name}_class_distribution.png")
print(f"  - {run_name}_efficientnet_comparison.png")
print(f"Logs:")
print(f"  - {run_name}_training_log.json")
print(f"  - {run_name}_training_log.csv")
print(f"  - {run_name}_classification_report.json")
print(f"  - {run_name}_final_summary.json")
print(f"  - {run_name}_summary.txt")
print(f"Models:")
print(f"  - {run_name}_best.pth")
print(f"{'='*60}")

#====================
# 15. Dataset Cleanup Utility (Optional)
#====================

def cleanup_corrupted_images(dataset_dir):
    """Remove corrupted images from dataset directory"""
    print(f"Scanning for corrupted images in {dataset_dir}...")
    
    corrupted_files = []
    extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.lower().endswith(extensions):
                file_path = os.path.join(root, file)
                try:
                    with Image.open(file_path) as img:
                        img.verify()
                    # Try to actually load it
                    with Image.open(file_path) as img:
                        img.convert('RGB')
                except Exception as e:
                    print(f"Corrupted: {file_path} - {e}")
                    corrupted_files.append(file_path)
    
    if corrupted_files:
        response = input(f"Found {len(corrupted_files)} corrupted files. Delete them? (y/n): ")
        if response.lower() == 'y':
            for file_path in corrupted_files:
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Failed to delete {file_path}: {e}")
        else:
            # Save list to file
            with open('corrupted_files_list.txt', 'w') as f:
                for file_path in corrupted_files:
                    f.write(f"{file_path}\n")
            print("Corrupted files list saved to 'corrupted_files_list.txt'")
    else:
        print("No corrupted images found!")

#====================
# Model Selection Guide
#====================
def print_model_selection_guide():
    """Print a guide for selecting EfficientNet models"""
    print(f"\n{'='*80}")
    print("EFFICIENTNET MODEL SELECTION GUIDE")
    print(f"{'='*80}")
    
    guide_data = [
        ("B0", "5.3M", "224", "Fastest training, good for prototyping", "Limited GPU memory"),
        ("B1", "7.8M", "240", "Good balance of speed and accuracy", "Most applications"),
        ("B2", "9.2M", "260", "Better accuracy, still reasonable speed", "Production systems"),
        ("B3", "12M", "300", "Higher accuracy, moderate resources", "High-quality results needed"),
        ("B4", "19M", "380", "Very high accuracy, more resources", "Research, competitions"),
        ("B5", "30M", "456", "Excellent accuracy, high memory needs", "Maximum accuracy priority"),
        ("B6", "43M", "528", "Top accuracy, significant resources", "Research with powerful GPUs"),
        ("B7", "66M", "600", "Highest accuracy, longest training", "State-of-the-art results")
    ]
    
    print(f"{'Version':<8}{'Params':<8}{'Input':<8}{'Description':<35}{'Best For'}")
    print("-" * 80)
    for version, params, input_size, desc, best_for in guide_data:
        marker = ">>> " if version == EFFICIENTNET_VERSION else "    "
        print(f"{marker}{version:<5}{params:<8}{input_size:<8}{desc:<35}{best_for}")
    
        print(f"\n{'='*80}")
        print(f"CURRENT SELECTION: EfficientNet-{EFFICIENTNET_VERSION}")
        print(f"To change model, modify EFFICIENTNET_VERSION variable at the top of the script")
        print(f"{'='*80}")

print_model_selection_guide()