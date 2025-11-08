# ONNX Runtime Integration Summary

## What Was Done

Integrated ONNX Runtime Web to allow the SafeView extension to use a PyTorch-trained EfficientNet-B0 model alongside the existing TensorFlow.js models. Users can now switch between runtimes via the extension popup UI.

## Key Changes

### 1. Model Conversion
- **Created** `reference_files/convert_to_onnx.py` - converts `.pth` (PyTorch state_dict) to ONNX format
- **Output** will be `src/assets/models/efficientnetb0/efficientnet_b0.onnx` (user must run conversion)
- Matches training preprocessing from `reference_files/train.py` (EfficientNet-B0, 224x224 input, ImageNet normalization)

### 2. Extension Code Changes

#### Settings & UI
- **`src/constants.js`**: Added `selectedModel: 'tfjs'` to `DEFAULT_SETTINGS`
- **`src/popup.html`**: Added "Detection Model" dropdown in Advanced Options (tfjs / onnx_b0)
- **`src/popup.js`**: 
  - Added default for `selectedModel` in `ensureDefaultSettings()`
  - Load/display/save `selectedModel` setting
  - Updated select handler to support string values (not just integers)
- **`src/background.js`**: Added `selectedModel: 'tfjs'` to `defaultSettings`
- **`src/modules/settings.js`**: 
  - Added `getSelectedModel()` method
  - Emit `changeModel` event when `selectedModel` changes

#### Runtime Switching
- **`src/modules/onnx_detector.js`** (NEW): 
  - `OnnxDetector` class wrapping ONNX Runtime Web
  - Preprocessing (resize, normalize) matching training code
  - API compatible with existing `Detector` class
- **`src/offscreen.js`**:
  - Import `OnnxDetector`
  - `loadModels()` checks `selectedModel` setting and loads appropriate runtime
  - `runDetection()` routes to TFJS or ONNX based on current setting
  - Listens for `changeModel` event to reload models on user switch
- **`src/offscreen.html`**: Added ONNX Runtime Web CDN script tag

### 3. Reference Files (in `reference_files/`)
- **`convert_to_onnx.py`**: Conversion script with model architecture matching training
- **`onnx_browser_example.js`**: Minimal reference example for ONNX Runtime usage
- **`onnx_test.html`**: Standalone test page to validate ONNX model inference
- **`README_convert_to_onnx.md`**: Detailed documentation for conversion, testing, troubleshooting

## How to Use

### Step 1: Convert Model to ONNX
```bash
# Install dependencies
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install torch torchvision

# Run conversion (from repo root)
python reference_files/convert_to_onnx.py
```

**Output**: `src/assets/models/efficientnetb0/efficientnet_b0.onnx`

### Step 2: Test ONNX Model (Optional)
```bash
# Serve test HTML
python -m http.server 8000
# Open http://localhost:8000/reference_files/onnx_test.html
# Select image → Run Inference
```

### Step 3: Use in Extension
1. Load extension in Chrome (Developer mode → Load unpacked)
2. Open extension popup
3. Go to **Advanced Options** (click to expand)
4. Change **Detection Model** dropdown:
   - `TFJS (default)` - existing TensorFlow.js models
   - `ONNX - EfficientNet B0` - new PyTorch-based model
5. Selection persists across sessions via `chrome.storage.sync`

## Architecture

```
User selects model in popup
    ↓
popup.js saves to chrome.storage.sync (selectedModel)
    ↓
Settings.updateSettings() emits 'changeModel' event
    ↓
offscreen.js listens for event and calls loadModels()
    ↓
loadModels() checks selectedModel:
  - 'tfjs' → load detector.initHuman() + detector.initNsfwModel()
  - 'onnx_b0' → load onnxDetector.init() + detector.initHuman() (for gender detection)
    ↓
runDetection() routes inference based on selectedModel:
  - 'tfjs' → TFJS tensor pipeline (existing)
  - 'onnx_b0' → ONNX preprocessing + inference (new)
```

## Technical Details

### ONNX Model
- **Architecture**: EfficientNet-B0 backbone + custom 3-layer classifier
- **Classes**: ['Drawing', 'Hentai', 'Neutral', 'Porn', 'Sexy']
- **Input**: [1, 3, 224, 224] float32 (CHW format, ImageNet normalized)
- **Output**: [1, 5] logits (apply softmax for probabilities)
- **Size**: ~20-30 MB

### Preprocessing Pipeline
1. Resize/crop image to 224×224 (canvas)
2. Extract RGB pixels (0-255)
3. Normalize: `(pixel/255 - mean) / std` where mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]
4. Convert to CHW float32 array
5. Create ONNX tensor and run inference

### Runtime Dependencies
- **ONNX Runtime Web**: Loaded via CDN (`https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js`)
- **Manifest**: No CSP changes needed (WASM allowed by default in extension offscreen documents)
- **Web Accessible Resources**: `src/assets/*` already declared in manifest.json

## Limitations & Future Work

### Current Limitations
- ONNX runtime only supports NSFW detection (no gender detection yet)
- Model must be converted manually by user (`.pth` → `.onnx`)
- ONNX model size larger than TFJS NSFW model

### Future Enhancements
- Add gender detection to ONNX runtime (train/convert face detection model)
- Support additional EfficientNet variants (B1-B7)
- Model quantization for smaller size
- Pre-convert model and include in releases
- Benchmark ONNX vs TFJS performance

## Files Added/Modified

### New Files
- `src/modules/onnx_detector.js`
- `reference_files/convert_to_onnx.py`
- `reference_files/onnx_browser_example.js`
- `reference_files/onnx_test.html`
- `reference_files/README_convert_to_onnx.md`
- `reference_files/ONNX_INTEGRATION_SUMMARY.md` (this file)

### Modified Files
- `src/constants.js` - added `selectedModel` default
- `src/background.js` - added `selectedModel` default
- `src/popup.html` - added model selector dropdown
- `src/popup.js` - load/save/display selectedModel setting
- `src/modules/settings.js` - added `getSelectedModel()` and `changeModel` event
- `src/offscreen.js` - runtime switching logic
- `src/offscreen.html` - ONNX Runtime CDN script

## Testing Checklist

- [ ] Run `convert_to_onnx.py` to generate ONNX model
- [ ] Test `onnx_test.html` with sample images
- [ ] Load extension and verify TFJS mode works (default)
- [ ] Switch to ONNX mode in popup Advanced Options
- [ ] Verify ONNX model loads in offscreen (check console)
- [ ] Test image detection with ONNX runtime
- [ ] Test video detection with ONNX runtime
- [ ] Switch back to TFJS and verify seamless transition
- [ ] Verify setting persists across browser restarts

## Troubleshooting

**Issue**: "ort is not defined"  
**Fix**: Ensure `offscreen.html` includes ONNX Runtime CDN script before `offscreen.js`

**Issue**: "Failed to load ONNX model"  
**Fix**: Run conversion script to generate `.onnx` file; verify path in console logs

**Issue**: Wrong predictions  
**Fix**: Verify preprocessing normalization matches training (ImageNet mean/std)

**Issue**: Extension crashes when switching models  
**Fix**: Check browser console for errors; may need to reload extension after first ONNX model load

---

For detailed documentation, see `reference_files/README_convert_to_onnx.md`.
