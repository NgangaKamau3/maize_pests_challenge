#!/usr/bin/env python
"""
Submission generator for Fall Armyworm Detection System.
This script runs inference on test images and generates a submission file.
"""

import os
import torch
import pandas as pd
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np

# Import custom modules
from preprocessing import get_transforms, TestDataset
from training import FallArmywormClassifier

def load_model(model_path, device='cuda'):
    """Load trained model from checkpoint"""
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU instead")
        device = 'cpu'
    
    try:
        # Try with weights_only=False (for PyTorch 2.6+)
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        # Fallback for older PyTorch versions
        checkpoint = torch.load(model_path, map_location=device)
    
    model = FallArmywormClassifier(
        model_name=checkpoint.get('model_name', 'mobilenet_v2'),
        pretrained=False
    )
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model loaded from {model_path}")
    if 'val_auc' in checkpoint:
        print(f"📊 Best validation AUC: {checkpoint.get('val_auc', 'N/A'):.4f}")
    
    return model

def predict(model, test_loader, device):
    """Run inference on test data"""
    model.eval()
    all_probs = []
    all_ids = []
    
    with torch.no_grad():
        for data, image_ids in tqdm(test_loader, desc="Inference"):
            data = data.to(device)
            outputs = model(data).squeeze()
            
            if outputs.dim() == 0:  # Single image case
                outputs = outputs.unsqueeze(0)
            
            probs = torch.sigmoid(outputs).cpu().numpy()
            
            all_probs.extend(probs)
            all_ids.extend(image_ids)
    
    return all_ids, all_probs

def create_submission(ids, probs, output_path='submission.csv'):
    """Create submission file in required format"""
    submission_df = pd.DataFrame({
        'Image_ID': ids,
        'Target': probs
    })
    
    submission_df.to_csv(output_path, index=False)
    
    print(f"💾 Submission saved to: {output_path}")
    print(f"📊 Total predictions: {len(submission_df)}")
    
    return submission_df

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Generate submission for Fall Armyworm Detection')
    parser.add_argument('--model_path', type=str, default='./models/best_model.pth',
                        help='Path to trained model checkpoint')
    parser.add_argument('--test_csv', type=str, 
                        default='./PestDataset/Combined_pestDataset/Test.csv',
                        help='Path to test CSV file')
    parser.add_argument('--images_dir', type=str,
                        default='./PestDataset/Combined_pestDataset/Images',
                        help='Path to images directory')
    parser.add_argument('--output_path', type=str, default='submission.csv',
                        help='Path to save submission CSV')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for inference')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (default: auto-detect)')
    return parser.parse_args()

def main():
    """Main function to generate submission"""
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
        'test_csv': args.test_csv,
        'images_dir': args.images_dir,
        'output_path': args.output_path,
        'batch_size': args.batch_size,
        'input_size': 224,
        'device': device,
        'num_workers': 4
    }
    
    print("🌾 Fall Armyworm Detection - Submission Generator")
    print("=" * 60)
    
    # Check if model exists
    if not os.path.exists(CONFIG['model_path']):
        print(f"❌ Model not found at {CONFIG['model_path']}")
        return
    
    # Load model
    model = load_model(CONFIG['model_path'], CONFIG['device'])
    
    # Load test data
    test_transforms = get_transforms(input_size=CONFIG['input_size'], augment=False)
    
    # Read test CSV
    test_df = pd.read_csv(CONFIG['test_csv'])
    test_image_ids = test_df['Image_id'].tolist()
    
    # Create test dataset
    test_dataset = TestDataset(
        image_dir=CONFIG['images_dir'],
        image_ids=test_image_ids,
        transform=test_transforms
    )
    
    # Set pin_memory based on device
    pin_memory = CONFIG['device'] == 'cuda' and torch.cuda.is_available()
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=CONFIG['num_workers'],
        pin_memory=pin_memory
    )
    
    print(f"📁 Loaded {len(test_dataset)} test images")
    
    # Run inference
    print("\n🔮 Running inference...")
    ids, probs = predict(model, test_loader, CONFIG['device'])
    
    # Create submission
    submission_df = create_submission(ids, probs, CONFIG['output_path'])
    
    print("\n✅ Submission file created successfully!")

if __name__ == "__main__":
    main()