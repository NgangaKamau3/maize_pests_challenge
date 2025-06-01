#!/usr/bin/env python
"""
Evaluation script for Fall Armyworm Detection System.
This script evaluates the trained model and generates performance metrics.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import cv2
import warnings
import argparse
warnings.filterwarnings('ignore')

# Import FallArmywormClassifier from training.py
from train import FallArmywormClassifier

class TestDataset(Dataset):
    """Dataset for test images without labels"""
    
    def __init__(self, image_dir, transform=None, valid_extensions=('.jpg', '.jpeg', '.png')):
        self.image_dir = Path(image_dir)
        self.transform = transform
        
        # Get all valid image files
        self.image_paths = []
        for ext in valid_extensions:
            self.image_paths.extend(list(self.image_dir.glob(f'*{ext}')))
            self.image_paths.extend(list(self.image_dir.glob(f'*{ext.upper()}'))
)
        
        self.image_paths = sorted(self.image_paths)
        print(f"📁 Found {len(self.image_paths)} test images")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        try:
            # Load and convert image
            image = Image.open(image_path).convert('RGB')
            
            # Apply transforms
            if self.transform:
                image = self.transform(image)
            
            # Return image and filename
            return image, image_path.name
        
        except Exception as e:
            print(f"⚠️ Error loading {image_path}: {e}")
            # Return a blank image if loading fails
            blank_image = Image.new('RGB', (224, 224), color='white')
            if self.transform:
                blank_image = self.transform(blank_image)
            return blank_image, image_path.name

def get_inference_transforms(input_size=224):
    """Get transforms for inference (same as validation)"""
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

def load_trained_model(model_path, model_class, device='cuda'):
    """Load trained model from checkpoint"""
    
    # Load checkpoint
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)
    
    # Initialize model
    model = model_class(
        model_name=checkpoint.get('model_name', 'mobilenet_v2'),
        pretrained=False  # We're loading trained weights
    )
    
    # Load trained weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model loaded from {model_path}")
    print(f"📊 Best validation AUC: {checkpoint.get('val_auc', 'N/A'):.4f}")
    print(f"🎯 Training epoch: {checkpoint.get('epoch', 'N/A')}")
    
    return model

def run_inference_on_test_set(model, test_dir, device='cuda', batch_size=32, input_size=224, save_dir='./results'):
    """Run inference on test images and generate predictions"""
    
    # Create test dataset and loader
    test_transforms = get_inference_transforms(input_size)
    test_dataset = TestDataset(test_dir, transform=test_transforms)
    
    if len(test_dataset) == 0:
        raise ValueError(f"No valid images found in {test_dir}")
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )
    
    print(f"🔮 Running inference on {len(test_dataset)} test images...")
    
    model.eval()
    predictions = []
    filenames = []
    probabilities = []
    
    with torch.no_grad():
        for batch_idx, (data, batch_filenames) in enumerate(tqdm(test_loader, desc='Inference')):
            data = data.to(device)
            
            # Get model outputs
            outputs = model(data).squeeze()
            
            # Convert to probabilities
            if outputs.dim() == 0:  # Single image case
                outputs = outputs.unsqueeze(0)
            
            probs = torch.sigmoid(outputs).cpu().numpy()
            
            # Convert to binary predictions (threshold = 0.5)
            batch_preds = (probs > 0.5).astype(int)
            
            # Store results
            predictions.extend(batch_preds)
            probabilities.extend(probs)
            filenames.extend(batch_filenames)
    
    print(f"✅ Inference completed on {len(predictions)} images")
    
    # Create results summary
    unique_preds, counts = np.unique(predictions, return_counts=True)
    print(f"\n📊 PREDICTION SUMMARY:")
    for pred, count in zip(unique_preds, counts):
        label = "Infected" if pred == 1 else "Healthy"
        percentage = (count / len(predictions)) * 100
        print(f"   {label}: {count} images ({percentage:.1f}%)")
    
    return predictions, probabilities, filenames

def create_submission_csv(predictions, filenames, output_path='submission.csv'):
    """Create submission CSV in required format"""
    
    # Create DataFrame
    submission_df = pd.DataFrame({
        'Image_ID': filenames,
        'Target': predictions
    })
    
    # Sort by ID for consistency
    submission_df = submission_df.sort_values('Image_ID').reset_index(drop=True)
    
    # Save to CSV
    submission_df.to_csv(output_path, index=False)
    
    print(f"💾 Submission CSV saved: {output_path}")
    print(f"📊 Total predictions: {len(submission_df)}")
    print(f"📋 Preview of submission:")
    print(submission_df.head(10))
    
    return submission_df

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate Fall Armyworm Detection Model')
    parser.add_argument('--model_path', type=str, default='./models/best_model.pth',
                        help='Path to trained model checkpoint')
    parser.add_argument('--test_dir', type=str, 
                        default='./PestDataset/Combined_pestDataset/Images',
                        help='Path to test images directory')
    parser.add_argument('--output_csv', type=str, default='submission.csv',
                        help='Path to save submission CSV')
    parser.add_argument('--save_dir', type=str, default='./results',
                        help='Directory to save evaluation results')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (default: auto-detect)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for inference')
    return parser.parse_args()

def main():
    """Main function"""
    # Parse arguments
    args = parse_args()
    
    # Set device
    if args.device:
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Configuration
    CONFIG = {
        'model_path': args.model_path,
        'test_dir': args.test_dir,
        'output_csv': args.output_csv,
        'save_dir': args.save_dir,
        'device': device,
        'batch_size': args.batch_size
    }
    
    print("🌾 Fall Armyworm Detection - Evaluation")
    print("=" * 60)
    
    # Create results directory
    Path(CONFIG['save_dir']).mkdir(exist_ok=True)
    
    try:
        # Load model
        model = load_trained_model(
            model_path=CONFIG['model_path'],
            model_class=FallArmywormClassifier,
            device=CONFIG['device']
        )
        
        # Run inference on test set
        predictions, probabilities, filenames = run_inference_on_test_set(
            model=model,
            test_dir=CONFIG['test_dir'],
            device=CONFIG['device'],
            batch_size=CONFIG['batch_size'],
            save_dir=CONFIG['save_dir']
        )
        
        # Create submission CSV
        submission_df = create_submission_csv(
            predictions=probabilities,  # Use probabilities for AUC metric
            filenames=filenames,
            output_path=CONFIG['output_csv']
        )
        
        print(f"🎉 SUCCESS!")
        print(f"📊 Generated predictions for {len(predictions)} images")
        print(f"💾 Submission saved: {CONFIG['output_csv']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("🔧 Please check file paths and model compatibility")

if __name__ == "__main__":
    main()