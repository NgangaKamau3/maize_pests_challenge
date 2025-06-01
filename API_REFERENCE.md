# Fall Armyworm Detection System: API Reference

## Overview

This document provides a comprehensive reference for the Fall Armyworm Detection System API. The system offers both Python API interfaces for direct integration and REST API endpoints for remote access.

## Python API

### Data Handling

#### `MaizeLeafDataset` Class

A PyTorch Dataset class for loading and preprocessing maize leaf images.

```python
from preprocessing import MaizeLeafDataset

dataset = MaizeLeafDataset(
    image_paths=['path/to/image1.jpg', 'path/to/image2.jpg'],
    labels=[0, 1],  # 0: Healthy, 1: Infected
    transform=transform_function
)
```

**Parameters:**
- `image_paths` (list): List of paths to image files
- `labels` (list): List of binary labels (0: Healthy, 1: Infected)
- `transform` (callable, optional): Transformation function to apply to images

**Returns:**
- PyTorch Dataset object that yields (image, label) tuples

#### `TestDataset` Class

A PyTorch Dataset class for loading test images without labels.

```python
from preprocessing import TestDataset

test_dataset = TestDataset(
    image_dir='path/to/images',
    transform=transform_function
)
```

**Parameters:**
- `image_dir` (str): Directory containing test images
- `transform` (callable, optional): Transformation function to apply to images

**Returns:**
- PyTorch Dataset object that yields (image, image_id) tuples

#### `create_data_loaders` Function

Creates training and validation data loaders from a CSV file.

```python
from preprocessing import create_data_loaders

train_loader, val_loader, class_counts = create_data_loaders(
    csv_path='path/to/train.csv',
    images_dir='path/to/images',
    batch_size=32,
    input_size=224,
    train_split=0.8
)
```

**Parameters:**
- `csv_path` (str): Path to CSV file with Image_id and Label columns
- `images_dir` (str): Directory containing images
- `batch_size` (int): Batch size for training
- `input_size` (int): Input image size
- `train_split` (float): Fraction of data for training
- `num_workers` (int): Number of worker processes for data loading
- `pin_memory` (bool): Whether to pin memory for faster GPU transfer

**Returns:**
- `train_loader`: DataLoader for training data
- `val_loader`: DataLoader for validation data
- `class_counts`: List with count of samples in each class

### Model

#### `FallArmywormClassifier` Class

The main model class for fall armyworm detection.

```python
from training import FallArmywormClassifier

model = FallArmywormClassifier(
    model_name='mobilenet_v2',
    pretrained=True,
    dropout=0.3
)
```

**Parameters:**
- `model_name` (str): Base model architecture ('mobilenet_v2' or 'resnet18')
- `pretrained` (bool): Whether to use pre-trained weights
- `dropout` (float): Dropout rate for regularization

**Methods:**
- `forward(x)`: Forward pass through the network
- `get_model_size()`: Calculate model size in MB

#### `train_model` Function

Trains the model with early stopping and metrics tracking.

```python
from training import train_model

trained_model, metrics, best_auc = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=50,
    learning_rate=0.001,
    device='cuda',
    save_dir='./models'
)
```

**Parameters:**
- `model`: FallArmywormClassifier instance
- `train_loader`: DataLoader for training data
- `val_loader`: DataLoader for validation data
- `num_epochs` (int): Maximum number of training epochs
- `learning_rate` (float): Initial learning rate
- `device` (str): Device to use ('cuda' or 'cpu')
- `save_dir` (str): Directory to save model checkpoints

**Returns:**
- `trained_model`: Trained model with best weights
- `metrics`: MetricsTracker instance with training history
- `best_auc`: Best validation AUC score achieved

#### `evaluate_model` Function

Evaluates the model and generates performance metrics.

```python
from training import evaluate_model

auc, accuracy, probs, labels = evaluate_model(
    model=trained_model,
    val_loader=val_loader,
    device='cuda',
    save_dir='./models'
)
```

**Parameters:**
- `model`: Trained FallArmywormClassifier instance
- `val_loader`: DataLoader for validation data
- `device` (str): Device to use ('cuda' or 'cpu')
- `save_dir` (str): Directory to save evaluation results

**Returns:**
- `auc`: AUC score
- `accuracy`: Classification accuracy
- `probs`: Predicted probabilities
- `labels`: True labels

### Mobile Optimization

#### `ModelOptimizer` Class

Handles model conversion and optimization for mobile deployment.

```python
from quantize import ModelOptimizer

optimizer = ModelOptimizer(
    model_path='./models/best_model.pth',
    input_size=(224, 224)
)

model = optimizer.load_model(backbone='mobilenet_v2')
onnx_path = optimizer.convert_to_onnx(model, output_path='model.onnx')
quantized_path = optimizer.quantize_onnx_model(onnx_path, quantized_path='model_quantized.onnx')
stats = optimizer.benchmark_model(quantized_path)
```

**Methods:**
- `load_model(backbone)`: Load trained PyTorch model
- `convert_to_onnx(model, output_path)`: Convert PyTorch model to ONNX format
- `quantize_onnx_model(onnx_path, quantized_path)`: Apply dynamic quantization
- `benchmark_model(model_path, num_runs)`: Benchmark model inference speed

#### `MobileInference` Class

Mobile-optimized inference class for deployment.

```python
from quantize import MobileInference

classifier = MobileInference(
    model_path='fall_armyworm_quantized.onnx',
    input_size=(224, 224)
)

result = classifier.predict('path/to/image.jpg')
print(f"Prediction: {result['prediction']}")
print(f"Probability: {result['probability']:.2f}")
print(f"Inference time: {result['inference_time_ms']:.1f}ms")
```

**Methods:**
- `preprocess_image(image_path)`: Preprocess image for inference
- `predict(image_path, threshold)`: Run inference on single image
- `batch_predict(image_paths, threshold)`: Run inference on multiple images

## REST API

The system can be deployed as a REST API service using Flask.

### Endpoints

#### `POST /api/predict`

Analyzes an image and returns fall armyworm detection results.

**Request:**
```
POST /api/predict
Content-Type: multipart/form-data

file: [image file]
```

**Response:**
```json
{
  "prediction": 1,
  "probability": 0.95,
  "confidence": 0.95,
  "inference_time_ms": 24.5,
  "status": "success"
}
```

#### `GET /api/models`

Lists available models.

**Request:**
```
GET /api/models
```

**Response:**
```json
{
  "models": [
    {
      "id": "mobilenet_v2",
      "name": "MobileNetV2",
      "size_mb": 1.4,
      "avg_inference_time_ms": 25.3
    },
    {
      "id": "resnet18",
      "name": "ResNet18",
      "size_mb": 2.8,
      "avg_inference_time_ms": 35.7
    }
  ]
}
```

#### `POST /api/batch_predict`

Analyzes multiple images and returns results for each.

**Request:**
```
POST /api/batch_predict
Content-Type: multipart/form-data

files: [image files]
```

**Response:**
```json
{
  "results": [
    {
      "filename": "image1.jpg",
      "prediction": 1,
      "probability": 0.95,
      "confidence": 0.95
    },
    {
      "filename": "image2.jpg",
      "prediction": 0,
      "probability": 0.12,
      "confidence": 0.88
    }
  ],
  "inference_time_ms": 58.3,
  "status": "success"
}
```

### Error Responses

**Invalid Request:**
```json
{
  "error": "No file provided",
  "status": "error"
}
```

**Server Error:**
```json
{
  "error": "Internal server error",
  "status": "error"
}
```

## Command-Line Interface

### Training Script

```bash
python train.py --epochs 50 --lr 0.001 --batch_size 32 --model mobilenet_v2 --no_early_stopping
```

**Arguments:**
- `--continue_training`: Continue training from saved checkpoint
- `--epochs`: Number of epochs to train (default: 50)
- `--lr`: Learning rate (default: 0.001)
- `--batch_size`: Batch size for training (default: 32)
- `--model`: Model architecture (default: mobilenet_v2)
- `--no_early_stopping`: Disable early stopping
- `--data_dir`: Path to dataset directory
- `--save_dir`: Directory to save models
- `--device`: Device to use (default: auto-detect)

### Evaluation Script

```bash
python evaluation.py --model_path ./models/best_model.pth --test_dir ./PestDataset/Combined_pestDataset/Images
```

**Arguments:**
- `--model_path`: Path to trained model checkpoint
- `--test_dir`: Path to test images directory
- `--output_csv`: Path to save submission CSV
- `--save_dir`: Directory to save evaluation results
- `--device`: Device to use (default: auto-detect)
- `--batch_size`: Batch size for inference

### Quantization Script

```bash
python quantize.py --model_path ./models/best_model.pth --backbone mobilenet_v2 --test_inference
```

**Arguments:**
- `--model_path`: Path to trained model checkpoint
- `--backbone`: Model backbone architecture
- `--output_dir`: Directory to save optimized models
- `--input_size`: Input image size
- `--benchmark`: Run benchmarking on optimized models
- `--test_inference`: Test inference on sample images

### Inference Script

```bash
python main.py --model_path ./models/best_model.pth --test_csv ./PestDataset/Combined_pestDataset/Test.csv
```

**Arguments:**
- `--model_path`: Path to trained model checkpoint
- `--test_csv`: Path to test CSV file
- `--images_dir`: Path to images directory
- `--output_path`: Path to save submission CSV
- `--batch_size`: Batch size for inference
- `--device`: Device to use (default: auto-detect)