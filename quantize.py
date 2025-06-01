#!/usr/bin/env python
"""
Model optimization script for Fall Armyworm Detection System.
This script handles model conversion to ONNX format and quantization for mobile deployment.
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import onnx
import onnxruntime as ort
import numpy as np
from PIL import Image
import time
import os
import argparse
from pathlib import Path

class FallArmywormClassifier(nn.Module):
    """Lightweight binary classifier for fall armyworm detection"""
    def __init__(self, backbone='mobilenet_v2', pretrained=True):
        super().__init__()
        
        if backbone == 'mobilenet_v2':
            self.backbone = models.mobilenet_v2(pretrained=pretrained)
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(1280, 1)  # Binary classification
            )
        elif backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            self.backbone.fc = nn.Linear(512, 1)
        
    def forward(self, x):
        return self.backbone(x)

class ModelOptimizer:
    """Handles model conversion and optimization for mobile deployment"""
    
    def __init__(self, model_path, input_size=(224, 224)):
        self.model_path = model_path
        self.input_size = input_size
        self.device = torch.device('cpu')  # Force CPU for mobile compatibility
        
    def load_model(self, backbone='mobilenet_v2'):
        """Load trained PyTorch model"""
        model = FallArmywormClassifier(backbone=backbone, pretrained=False)
        
        # Load trained weights with PyTorch 2.6 compatibility
        try:
            # Try with weights_only=False for PyTorch 2.6+
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        except TypeError:
            # Fallback for older PyTorch versions
            checkpoint = torch.load(self.model_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        return model
    
    def convert_to_onnx(self, model, output_path='fall_armyworm_model.onnx'):
        """Convert PyTorch model to ONNX format"""
        print("Converting model to ONNX format...")
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, *self.input_size)
        
        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        # Verify ONNX model
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        
        print(f"✅ ONNX model saved to: {output_path}")
        return output_path
    
    def quantize_onnx_model(self, onnx_path, quantized_path='fall_armyworm_quantized.onnx'):
        """Apply dynamic quantization to ONNX model"""
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            
            print("Applying dynamic quantization...")
            quantize_dynamic(
                onnx_path,
                quantized_path,
                weight_type=QuantType.QUInt8
            )
            print(f"✅ Quantized model saved to: {quantized_path}")
            return quantized_path
        except ImportError:
            print("⚠️ ONNX quantization not available. Install: pip install onnxruntime")
            return onnx_path
    
    def benchmark_model(self, model_path, num_runs=100):
        """Benchmark model inference speed"""
        print(f"\n🔥 Benchmarking: {model_path}")
        
        # Create session
        session = ort.InferenceSession(model_path)
        
        # Prepare input
        dummy_input = np.random.randn(1, 3, *self.input_size).astype(np.float32)
        
        # Warmup
        for _ in range(10):
            session.run(None, {'input': dummy_input})
        
        # Benchmark
        start_time = time.time()
        for _ in range(num_runs):
            outputs = session.run(None, {'input': dummy_input})
        end_time = time.time()
        
        avg_latency = (end_time - start_time) / num_runs * 1000  # ms
        
        # Get model size
        model_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        
        print(f"📊 Model size: {model_size:.2f} MB")
        print(f"⚡ Average latency: {avg_latency:.2f} ms")
        print(f"🚀 Throughput: {1000/avg_latency:.1f} FPS")
        
        return {
            'size_mb': model_size,
            'latency_ms': avg_latency,
            'throughput_fps': 1000/avg_latency
        }

class MobileInference:
    """Mobile-optimized inference class"""
    
    def __init__(self, model_path, input_size=(224, 224)):
        self.input_size = input_size
        self.session = ort.InferenceSession(model_path)
        
        # Image preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_image(self, image_path):
        """Preprocess image for inference"""
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path  # Already PIL Image
            
        # Apply transforms
        tensor = self.transform(image)
        batch = tensor.unsqueeze(0)  # Add batch dimension
        
        return batch.numpy().astype(np.float32)
    
    def predict(self, image_path, threshold=0.5):
        """Run inference on single image"""
        # Preprocess
        input_tensor = self.preprocess_image(image_path)
        
        # Inference
        start_time = time.time()
        outputs = self.session.run(None, {'input': input_tensor})
        inference_time = time.time() - start_time
        
        # Get prediction
        logit = outputs[0][0][0]  # Extract scalar logit
        probability = torch.sigmoid(torch.tensor(logit)).item()
        prediction = int(probability >= threshold)
        
        return {
            'prediction': prediction,
            'probability': probability,
            'confidence': max(probability, 1 - probability),
            'inference_time_ms': inference_time * 1000
        }
    
    def batch_predict(self, image_paths, threshold=0.5):
        """Run inference on multiple images"""
        results = []
        
        for img_path in image_paths:
            try:
                result = self.predict(img_path, threshold)
                result['image_path'] = img_path
                results.append(result)
            except Exception as e:
                print(f"❌ Error processing {img_path}: {e}")
                
        return results

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Optimize Fall Armyworm Detection Model for Mobile')
    parser.add_argument('--model_path', type=str, default='./models/best_model.pth',
                        help='Path to trained model checkpoint')
    parser.add_argument('--backbone', type=str, default='mobilenet_v2',
                        choices=['mobilenet_v2', 'resnet18'],
                        help='Model backbone architecture')
    parser.add_argument('--output_dir', type=str, default='./',
                        help='Directory to save optimized models')
    parser.add_argument('--input_size', type=int, default=224,
                        help='Input image size')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run benchmarking on optimized models')
    parser.add_argument('--test_inference', action='store_true',
                        help='Test inference on sample images')
    return parser.parse_args()

def optimize_model_for_mobile(model_path, backbone='mobilenet_v2', output_dir='./', input_size=224):
    """Complete optimization pipeline"""
    print("🚀 Starting mobile optimization pipeline...")
    
    # Initialize optimizer
    optimizer = ModelOptimizer(model_path, input_size=(input_size, input_size))
    
    # Load model
    model = optimizer.load_model(backbone=backbone)
    print(f"✅ Loaded {backbone} model")
    
    # Convert to ONNX
    onnx_path = os.path.join(output_dir, 'fall_armyworm_model.onnx')
    onnx_path = optimizer.convert_to_onnx(model, onnx_path)
    
    # Quantize model
    quantized_path = os.path.join(output_dir, 'fall_armyworm_quantized.onnx')
    quantized_path = optimizer.quantize_onnx_model(onnx_path, quantized_path)
    
    # Benchmark both models
    print("\n" + "="*50)
    print("📈 PERFORMANCE COMPARISON")
    print("="*50)
    
    original_stats = optimizer.benchmark_model(onnx_path)
    quantized_stats = optimizer.benchmark_model(quantized_path)
    
    # Calculate improvements
    size_reduction = (1 - quantized_stats['size_mb'] / original_stats['size_mb']) * 100
    speed_improvement = (original_stats['latency_ms'] / quantized_stats['latency_ms'] - 1) * 100
    
    print(f"\n🎯 OPTIMIZATION RESULTS:")
    print(f"📦 Size reduction: {size_reduction:.1f}%")
    print(f"⚡ Speed improvement: {speed_improvement:.1f}%")
    
    return {
        'original_model': onnx_path,
        'optimized_model': quantized_path,
        'stats': {
            'original': original_stats,
            'quantized': quantized_stats
        }
    }

def mobile_app_inference_example(model_path):
    """Example of how to integrate in mobile app"""
    
    if not os.path.exists(model_path):
        print("⚠️ Model not found. Run optimization pipeline first.")
        return
    
    mobile_classifier = MobileInference(model_path)
    
    # Example: Process single image
    print("\n📱 MOBILE APP INFERENCE EXAMPLE")
    print("-" * 40)
    
    # Simulate camera capture or gallery selection
    sample_images = ['./PestDataset/Combined_pestDataset/Images/id_00exusbkgzw1b.jpg', 
                     './PestDataset/Combined_pestDataset/Images/id_046yl0cxn3ybz.jpg']
    
    for img_path in sample_images:
        if os.path.exists(img_path):
            result = mobile_classifier.predict(img_path)
            
            status = "🐛 INFECTED" if result['prediction'] == 1 else "✅ HEALTHY"
            
            print(f"Image: {img_path}")
            print(f"Status: {status}")
            print(f"Confidence: {result['confidence']:.1%}")
            print(f"Inference time: {result['inference_time_ms']:.1f}ms")
            print("-" * 40)

def main():
    """Main function"""
    # Parse arguments
    args = parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"❌ Model file not found: {args.model_path}")
        print("💡 Please run train.py first to generate the model.")
        return
    
    # Create output directory if it doesn't exist
    Path(args.output_dir).mkdir(exist_ok=True)
    
    # Run optimization pipeline
    results = optimize_model_for_mobile(
        model_path=args.model_path,
        backbone=args.backbone,
        output_dir=args.output_dir,
        input_size=args.input_size
    )
    
    # Test mobile inference if requested
    if args.test_inference:
        mobile_app_inference_example(results['optimized_model'])
    
    print("\n✅ Model optimization completed successfully!")
    print(f"📱 Original ONNX model: {results['original_model']}")
    print(f"📱 Quantized ONNX model: {results['optimized_model']}")

if __name__ == "__main__":
    main()