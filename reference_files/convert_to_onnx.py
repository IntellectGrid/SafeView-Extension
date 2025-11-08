"""
Reference conversion script:
- Loads the PyTorch .pth state_dict for the NSFW EfficientNet-B0 classifier
- Recreates the same model architecture (minimal NSFWClassifier) and loads weights
- Exports to ONNX at src/assets/models/efficientnetb0/efficientnet_b0.onnx

Usage:
    python reference_files/convert_to_onnx.py 

Notes:
- Run this in a Python environment with torch and torchvision installed.
- The script exports a model with input shape (1,3,224,224) and uses opset 13.
- Preprocessing used during training (and required at inference) is:
    - Resize -> CenterCrop to 224
    - Convert to tensor and Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])

"""
import os
import torch
import torch.nn as nn
from torchvision import models

# Paths
BASE = os.path.dirname(os.path.dirname(__file__))  # repo root
PTH_PATH = os.path.join(BASE, 'src', 'assets', 'models', 'efficientnetb0', 'efficientnet_b0_20250924_055016_best.pth')
OUT_DIR = os.path.join(BASE, 'src', 'assets', 'models', 'efficientnetb0')
OUT_ONNX = os.path.join(OUT_DIR, 'efficientnet_b0.onnx')

# Model parameters (must match training config)
EFFICIENTNET_VERSION = 'B0'
NUM_CLASSES = 5
INPUT_SIZE = 224

class NSFWClassifier(nn.Module):
    def __init__(self, efficientnet_version='B0', num_classes=5, pretrained=True):
        super().__init__()
        # Build backbone using torchvision's EfficientNet API
        if efficientnet_version == 'B0':
            backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        else:
            raise ValueError('Only B0 supported in this conversion script')

        self.features = backbone.features
        self.avgpool = backbone.avgpool
        in_feats = backbone.classifier[1].in_features

        # Same classifier as training code
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

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.exists(PTH_PATH):
        print(f"ERROR: .pth file not found at {PTH_PATH}")
        return

    # Instantiate model and load weights
    model = NSFWClassifier(efficientnet_version=EFFICIENTNET_VERSION, num_classes=NUM_CLASSES, pretrained=False)
    state = torch.load(PTH_PATH, map_location='cpu')
    # If saved with "model.state_dict()" it should be compatible, otherwise adapt
    try:
        model.load_state_dict(state)
    except RuntimeError as e:
        # try to handle wrapped state dicts
        if isinstance(state, dict) and 'state_dict' in state:
            model.load_state_dict(state['state_dict'])
        else:
            raise e

    model.eval()

    # Dummy input matching validation preprocessing (after Resize+CenterCrop)
    dummy_input = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE, dtype=torch.float32)

    # Export to ONNX
    print(f"Exporting ONNX to {OUT_ONNX} (opset 13)")
    torch.onnx.export(
        model,
        dummy_input,
        OUT_ONNX,
        input_names=['input'],
        output_names=['output'],
        opset_version=13,
        do_constant_folding=True,
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )

    print('ONNX export complete')

if __name__ == '__main__':
    main()
