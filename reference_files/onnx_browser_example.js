// Minimal example showing how to load an ONNX model using ONNX Runtime Web (ORT Web)
// This file is a reference and not yet wired into the extension.

// Include ORT Web via script tag in an HTML page (CDN):
// <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js"></script>

async function loadOnnxModel(onnxUrl) {
    if (typeof ort === 'undefined') {
        throw new Error('ort (onnxruntime-web) not found. Include ort.min.js in the page.');
    }

    // Initialize session
    const session = await ort.InferenceSession.create(onnxUrl);
    return session;
}

// Preprocess in browser: resize canvas, getImageData, convert to float32, normalize
function preprocessImageToTensor(imageData, inputSize) {
    // imageData: ImageData from canvas
    // Returns Float32Array in CHW order normalized with ImageNet mean/std
    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];
    const [width, height] = [imageData.width, imageData.height];
    const data = imageData.data; // RGBA

    // Simple center-crop + resize should be done prior (use canvas drawImage with cropping)
    // Assuming imageData is already RGB image with size inputSize x inputSize

    const floatData = new Float32Array(3 * inputSize * inputSize);
    let ptr = 0;
    for (let c = 0; c < 3; c++) {
        for (let y = 0; y < inputSize; y++) {
            for (let x = 0; x < inputSize; x++) {
                const idx = (y * inputSize + x) * 4;
                let value = data[idx + c] / 255.0;
                value = (value - mean[c]) / std[c];
                floatData[ptr++] = value;
            }
        }
    }
    return floatData;
}

async function runModel(session, preprocessedFloatData) {
    // session: ort.InferenceSession
    // preprocessedFloatData: Float32Array CHW
    const tensor = new ort.Tensor('float32', preprocessedFloatData, [1, 3, /*inputSize*/224, /*inputSize*/224]);
    const feeds = { input: tensor };
    const output = await session.run(feeds);
    return output.output.data; // Float32Array of logits
}

// Export functions for reference
if (typeof module !== 'undefined') {
    module.exports = { loadOnnxModel, preprocessImageToTensor, runModel };
}
