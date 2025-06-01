import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pathlib import Path
import random

class MaizeLeafDataset(Dataset):
    """Custom dataset for maize leaf images with fall armyworm classification"""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.long)

class TestDataset(Dataset):
    """Dataset for test images without labels"""
    
    def __init__(self, image_dir, image_ids=None, transform=None):
        self.image_dir = Path(image_dir)
        self.transform = transform
        
        if image_ids:
            # Use provided image IDs
            self.image_ids = image_ids
        else:
            # Get all image files in directory
            valid_extensions = ('.jpg', '.jpeg', '.png')
            self.image_ids = []
            for ext in valid_extensions:
                self.image_ids.extend([f.name for f in self.image_dir.glob(f'*{ext}')])
                self.image_ids.extend([f.name for f in self.image_dir.glob(f'*{ext.upper()}')])
            
            self.image_ids = sorted(self.image_ids)
        
        print(f"📁 Found {len(self.image_ids)} test images")
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = self.image_dir / image_id
        
        try:
            # Load and convert image
            image = Image.open(image_path).convert('RGB')
            
            # Apply transforms
            if self.transform:
                image = self.transform(image)
            
            # Return image and filename
            return image, image_id
        
        except Exception as e:
            print(f"⚠️ Error loading {image_path}: {e}")
            # Return a blank image if loading fails
            blank_image = Image.new('RGB', (224, 224), color='white')
            if self.transform:
                blank_image = self.transform(blank_image)
            return blank_image, image_id

def load_dataset_from_csv(csv_path, images_dir):
    """Load dataset from CSV file with Image_id and Label columns"""
    df = pd.read_csv(csv_path)
    
    # Extract image paths and labels
    image_paths = [os.path.join(images_dir, img_id) for img_id in df['Image_id']]
    labels = df['Label'].values
    
    print(f"Total images loaded: {len(image_paths)}")
    print(f"Class distribution - Healthy: {sum(labels==0)}, Infected: {sum(labels==1)}")
    
    return image_paths, labels

def get_transforms(input_size=224, augment=True):
    """
    Get image transforms for training and validation
    Optimized for mobile deployment with efficient augmentations
    """
    
    # Base transforms (always applied)
    base_transforms = [
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ]
    
    if augment:
        # Training transforms with augmentation
        train_transforms = transforms.Compose([
            transforms.Resize((int(input_size * 1.1), int(input_size * 1.1))),  # Slightly larger for crop
            transforms.RandomCrop(input_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),  # Leaves can be oriented differently
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # Add random erasing for better generalization
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.15), ratio=(0.3, 3.3))
        ])
        
        val_transforms = transforms.Compose(base_transforms)
        
        return train_transforms, val_transforms
    else:
        # Only validation transforms
        val_transforms = transforms.Compose(base_transforms)
        return val_transforms

def create_data_loaders(csv_path, images_dir, batch_size=32, input_size=224, 
                        train_split=0.8, num_workers=4, pin_memory=None):
    """
    Create training and validation DataLoaders from CSV file
    
    Args:
        csv_path: Path to CSV file with Image_id and Label columns
        images_dir: Path to directory containing images
        batch_size: Batch size for training
        input_size: Input image size (224 for mobile optimization)
        train_split: Fraction of data for training (0.8 = 80%)
        num_workers: Number of worker processes for data loading
        pin_memory: Pin memory for faster GPU transfer (auto-detected if None)
    
    Returns:
        train_loader, val_loader, class_counts
    """
    # Auto-detect pin_memory based on CUDA availability
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    
    # Load dataset
    image_paths, labels = load_dataset_from_csv(csv_path, images_dir)
    
    # Split data maintaining class balance
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, 
        train_size=train_split, 
        stratify=labels,  # Maintain class balance
        random_state=42
    )
    
    print(f"Training set: {len(train_paths)} images")
    print(f"Validation set: {len(val_paths)} images")
    
    # Get transforms
    train_transforms, val_transforms = get_transforms(input_size=input_size, augment=True)
    
    # Create datasets
    train_dataset = MaizeLeafDataset(train_paths, train_labels, transform=train_transforms)
    val_dataset = MaizeLeafDataset(val_paths, val_labels, transform=val_transforms)
    
    # Create data loaders with appropriate pin_memory setting
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Ensures consistent batch sizes
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    # Calculate class counts for potential loss weighting
    class_counts = [sum(labels==0), sum(labels==1)]
    
    return train_loader, val_loader, class_counts

def get_class_weights(class_counts):
    """
    Calculate class weights for handling class imbalance
    """
    total = sum(class_counts)
    weights = [total / (2 * count) for count in class_counts]
    return torch.tensor(weights, dtype=torch.float32)

# Example usage
if __name__ == "__main__":
    # Configuration
    CONFIG = {
        'csv_path': './PestDataset/Combined_pestDataset/Train.csv',
        'images_dir': './PestDataset/Combined_pestDataset/Images',
        'batch_size': 32,
        'input_size': 224,
        'train_split': 0.8,
        'num_workers': 4,
        # Auto-detect if CUDA is available for pin_memory
        'pin_memory': torch.cuda.is_available()
    }
    
    # Create data loaders
    try:
        train_loader, val_loader, class_counts = create_data_loaders(**CONFIG)
        
        print("\n=== Dataset Summary ===")
        print(f"Training batches: {len(train_loader)}")
        print(f"Validation batches: {len(val_loader)}")
        print(f"Class counts: {class_counts}")
        
        # Calculate class weights for potential loss balancing
        class_weights = get_class_weights(class_counts)
        print(f"Recommended class weights: {class_weights}")
        
        # Test data loading
        print("\n=== Testing Data Loading ===")
        train_batch = next(iter(train_loader))
        val_batch = next(iter(val_loader))
        
        print(f"Training batch shape: {train_batch[0].shape}")
        print(f"Training labels shape: {train_batch[1].shape}")
        print(f"Sample training labels: {train_batch[1][:8]}")
        
        print(f"Validation batch shape: {val_batch[0].shape}")
        print(f"Validation labels shape: {val_batch[1].shape}")
        print(f"Sample validation labels: {val_batch[1][:8]}")
        
        print("\n✅ Data pipeline successfully created and tested!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nPlease ensure:")
        print("1. Update paths to your dataset location")
        print("2. Images are in the correct directory")
        print("3. CSV file has the correct format (Image_id, Label columns)")