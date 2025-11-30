import onnxruntime as ort
import numpy as np
from PIL import Image
import argparse

def preprocess_image(image_path, input_size=224):
    """
    Load and preprocess image to match training transforms
    Args:
        image_path: Path to input image
        input_size: Model input size (224 for B0, 240 for B1, etc.)
    Returns:
        Preprocessed numpy array ready for ONNX inference
    """
    # Load image
    img = Image.open(image_path).convert('RGB')
    
    # Resize with aspect ratio preservation (like Resize + CenterCrop in training)
    # Add 32 pixels for better crop (matches val_transforms)
    resize_size = input_size + 32
    img.thumbnail((resize_size, resize_size), Image.Resampling.LANCZOS)
    
    # Center crop to exact input size
    width, height = img.size
    left = (width - input_size) // 2
    top = (height - input_size) // 2
    right = left + input_size
    bottom = top + input_size
    img = img.crop((left, top, right, bottom))
    
    # Convert to numpy array and normalize
    img_array = np.array(img, dtype=np.float32) / 255.0
    
    # Apply ImageNet normalization (same as training)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_array = (img_array - mean) / std
    
    # Convert from HWC to CHW format (PyTorch/ONNX format)
    img_array = np.transpose(img_array, (2, 0, 1))
    
    # Add batch dimension: (C, H, W) -> (1, C, H, W)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def run_inference(onnx_model_path, image_path, input_size=224):
    """
    Run inference on ONNX model
    Args:
        onnx_model_path: Path to .onnx model file
        image_path: Path to input image
        input_size: Model input size (default 224 for B0)
    Returns:
        predictions: Raw model outputs (logits)
        probabilities: Softmax probabilities
        predicted_class: Index of predicted class
    """
    # Load ONNX model
    print(f"Loading ONNX model from: {onnx_model_path}")
    session = ort.InferenceSession(onnx_model_path)
    
    # Get model input/output info
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    print(f"Model input name: {input_name}")
    print(f"Model output name: {output_name}")
    print(f"Expected input shape: {session.get_inputs()[0].shape}")
    
    # Preprocess image
    print(f"\nPreprocessing image: {image_path}")
    input_data = preprocess_image(image_path, input_size)
    print(f"Input shape: {input_data.shape}")
    
    # Run inference
    print("\nRunning inference...")
    outputs = session.run([output_name], {input_name: input_data})
    predictions = outputs[0][0]  # Remove batch dimension
    
    # Apply softmax to get probabilities
    exp_preds = np.exp(predictions - np.max(predictions))  # Numerical stability
    probabilities = exp_preds / np.sum(exp_preds)
    
    # Get predicted class
    predicted_class = np.argmax(probabilities)
    
    return predictions, probabilities, predicted_class

def main():
    parser = argparse.ArgumentParser(description="Test ONNX model inference")
    parser.add_argument("--model", type=str, default="model.onnx", 
                        help="Path to ONNX model file")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to input image")
    parser.add_argument("--input-size", type=int, default=224,
                        help="Model input size (224 for B0, 240 for B1, etc.)")
    args = parser.parse_args()
    
    # Class names from your training
    CLASSES = ['Drawing', 'Hentai', 'Neutral', 'Porn', 'Sexy']
    
    try:
        # Run inference
        predictions, probabilities, predicted_class = run_inference(
            args.model, 
            args.image, 
            args.input_size
        )
        
        # Display results
        print("\n" + "="*60)
        print("INFERENCE RESULTS")
        print("="*60)
        print(f"Predicted Class: {CLASSES[predicted_class]}")
        print(f"Confidence: {probabilities[predicted_class]:.2%}")
        print("\nAll class probabilities:")
        print("-"*60)
        
        # Sort by probability (descending)
        sorted_indices = np.argsort(probabilities)[::-1]
        for idx in sorted_indices:
            print(f"  {CLASSES[idx]:<15}: {probabilities[idx]:.2%} {'★' if idx == predicted_class else ''}")
        
        print("="*60)
        print(f"\nRaw logits: {predictions}")
        print(f"Probabilities: {probabilities}")
        
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        print("Make sure the model and image paths are correct.")
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()