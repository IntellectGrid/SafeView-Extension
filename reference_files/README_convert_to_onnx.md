# ONNX Runtime Integration - EfficientNet B0 NSFW Classifier

This folder contains reference files and scripts for converting the PyTorch `.pth` model to ONNX and integrating ONNX Runtime into the SafeView extension.

## Files

- **convert_to_onnx.py** - Python script to export PyTorch model to ONNX format
- **onnx_browser_example.js** - Minimal browser example for ONNX Runtime Web usage (reference)
- **onnx_test.html** - Standalone HTML page to test ONNX model loading and inference
- **README_convert_to_onnx.md** - This file

## How to Convert (.pth → .onnx)

1. **Create a Python environment** with torch and torchvision installed (torch >= 1.12 recommended):

   ```bash
   python -m venv venv
   # Activate venv (Windows: venv\Scripts\activate; Unix: source venv/bin/activate)
   pip install torch torchvision
   ```

2. **Run the conversion script** from the repository root:

   ```bash
   python reference_files/convert_to_onnx.py
   ```

3. **Output**: ONNX file will be written to `src/assets/models/efficientnetb0/efficientnet_b0.onnx`.

## Preprocessing Notes

The training/validation pipeline used these transforms:
- Resize(INPUT_SIZE + 32) → CenterCrop(INPUT_SIZE) → ToTensor() → Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])

**Critical**: Ensure you apply the same normalization before feeding images into the ONNX model in the browser.

## ONNX Runtime in Browser

- The extension uses **ONNX Runtime Web** (ort-web) loaded via CDN in `offscreen.html`.
- The `modules/onnx_detector.js` module handles ONNX model loading and inference with preprocessing matching the training code.
- See `onnx_test.html` for a minimal standalone test harness.

## Testing the ONNX Model

1. **Generate the ONNX model** (run `convert_to_onnx.py` as described above).
2. **Open `reference_files/onnx_test.html`** in a browser (you may need to serve it via a local server due to CORS restrictions on file:// protocol).
3. **Select a test image** and click "Run Inference" to see predictions (Drawing, Hentai, Neutral, Porn, Sexy).

Example local server (Python):

```bash
# From repo root
python -m http.server 8000
# Open http://localhost:8000/reference_files/onnx_test.html
```

## Extension Integration

### UI
- In the extension popup, go to **Advanced Options** → **Detection Model** and select:
  - `TFJS (default)` - uses existing TensorFlow.js models
  - `ONNX - EfficientNet B0` - uses the newly converted ONNX model

### Under the Hood
- Settings are persisted in `chrome.storage.sync` under key `selectedModel`.
- `src/offscreen.js` detects the selected model and loads the appropriate runtime.
- `modules/onnx_detector.js` provides a compatible API matching the TFJS detector.
- When the user switches models, the offscreen script reloads models dynamically.

### Manifest Changes
- No additional CSP changes required (WASM allowed by default in offscreen docs).
- `src/assets/*` is already web accessible in `manifest.json`.

## Model Sizes & Performance

- **ONNX model size**: ~20-30 MB (EfficientNet-B0 with custom classifier).
- **TFJS NSFW model size**: Varies (check `src/assets/models/NSFWX/`).
- **ONNX Runtime Web** uses WebAssembly for efficient inference.
- Performance may vary based on browser and hardware (GPU/WASM acceleration).

## Troubleshooting

- **"ort is not defined"**: Ensure `offscreen.html` includes the ONNX Runtime CDN script before `offscreen.js`.
- **Model not found**: Verify ONNX file exists at `src/assets/models/efficientnetb0/efficientnet_b0.onnx`.
- **Incorrect predictions**: Double-check preprocessing (normalization) matches training code.
- **CORS errors in test HTML**: Serve `onnx_test.html` via local HTTP server.

## Future Enhancements

- Add gender detection support to ONNX runtime (currently only NSFW detection).
- Support additional EfficientNet variants (B1-B7) if trained models available.
- Optimize ONNX model (quantization, pruning) for smaller size and faster inference.

---

For questions or issues, refer to the main project README or check browser console logs for detailed error messages.

