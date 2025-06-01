#!/usr/bin/env python
"""
Training script for Fall Armyworm Detection System.
This script handles the complete training pipeline including data loading,
model training, evaluation, and export for mobile deployment.
"""

import os
import torch
import argparse
from pathlib import Path

# Import custom modules
from preprocessing import create_data_loaders
from training import FallArmywormClassifier, train_model, evaluate_model, export_for_mobile

def custom_train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001, 
                device='cuda', save_dir='./models', disable_early_stopping=False):
    """Modified version of train_model that works with older PyTorch versions"""
    import torch.nn as nn
    import torch.optim as optim
    import time
    import copy
    from tqdm import tqdm
    import numpy as np
    from training import calculate_auc, MetricsTracker, EarlyStopping
    
    # Setup
    Path(save_dir).mkdir(exist_ok=True)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
    
    # Early stopping and metrics tracking
    early_stopping = EarlyStopping(patience=10, min_delta=0.001) if not disable_early_stopping else None
    metrics = MetricsTracker()
    
    # Move model to device
    model = model.to(device)
    
    print(f"🚀 Starting training on {device}")
    print(f"📱 Model size: {model.get_model_size():.2f} MB")
    print(f"🎯 Target: Mobile-optimized fall armyworm detection")
    if disable_early_stopping:
        print("⚠️ Early stopping disabled - will train for all epochs")
    print("-" * 60)
    
    best_val_auc = 0.0
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Training phase
        model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(train_loader, desc='Training')
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device).float()
            
            optimizer.zero_grad()
            outputs = model(data).squeeze()
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Collect predictions for AUC
            probs = torch.sigmoid(outputs).cpu().detach().numpy()
            all_preds.extend(probs)
            all_labels.extend(target.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        train_loss = running_loss / len(train_loader)
        train_auc = calculate_auc(all_labels, all_preds)
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validation')
            for data, target in pbar:
                data, target = data.to(device), target.to(device).float()
                
                outputs = model(data).squeeze()
                loss = criterion(outputs, target)
                running_loss += loss.item()
                
                # Collect predictions for AUC
                probs = torch.sigmoid(outputs).cpu().numpy()
                all_preds.extend(probs)
                all_labels.extend(target.cpu().numpy())
                
                pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        val_loss = running_loss / len(val_loader)
        val_auc = calculate_auc(all_labels, all_preds)
        
        # Learning rate scheduling
        scheduler.step(val_auc)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update metrics
        metrics.update(train_loss, val_loss, train_auc, val_auc, current_lr)
        
        # Logging
        epoch_time = time.time() - epoch_start
        print(f'Epoch [{epoch+1:2d}/{num_epochs}] | '
              f'Train Loss: {train_loss:.4f} | Train AUC: {train_auc:.4f} | '
              f'Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f} | '
              f'LR: {current_lr:.6f} | Time: {epoch_time:.1f}s')
        
        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': val_auc,
                'model_name': model.model_name
            }, f'{save_dir}/best_model.pth')
            print(f'💾 New best model saved! Val AUC: {val_auc:.4f}')
        
        # Early stopping check
        if not disable_early_stopping and early_stopping(val_auc, model):
            print(f'🛑 Early stopping triggered after {epoch+1} epochs')
            break
    
    # Restore best weights if early stopping was used
    if not disable_early_stopping:
        early_stopping.restore(model)
    
    training_time = time.time() - start_time
    print(f"\n✅ Training completed!")
    print(f"⏱️  Total time: {training_time/60:.1f} minutes")
    print(f"🏆 Best validation AUC: {best_val_auc:.4f}")
    
    # Plot training metrics
    metrics.plot_metrics(save_path=f'{save_dir}/training_metrics.png')
    
    return model, metrics, best_val_auc

def continue_training(model_path, train_loader, val_loader, additional_epochs=20, 
                     learning_rate=0.0001, device='cuda', save_dir='./models'):
    """Continue training from a saved checkpoint"""
    # Load the model checkpoint
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)
    
    # Initialize model
    model = FallArmywormClassifier(
        model_name=checkpoint.get('model_name', 'mobilenet_v2'),
        pretrained=False
    )
    
    # Load model weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    print(f"🔄 Resuming training from epoch {checkpoint.get('epoch', 0) + 1}")
    print(f"📊 Previous best AUC: {checkpoint.get('val_auc', 'N/A'):.4f}")
    
    # Train with early stopping disabled
    return custom_train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=additional_epochs,
        learning_rate=learning_rate,
        device=device,
        save_dir=save_dir,
        disable_early_stopping=True  # Disable early stopping for continued training
    )

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Fall Armyworm Detection Model')
    parser.add_argument('--continue_training', action='store_true', 
                        help='Continue training from saved checkpoint')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001, 0.0001 for continued training)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training (default: 32)')
    parser.add_argument('--model', type=str, default='mobilenet_v2',
                        choices=['mobilenet_v2', 'resnet18'],
                        help='Model architecture to use (default: mobilenet_v2)')
    parser.add_argument('--no_early_stopping', action='store_true',
                        help='Disable early stopping')
    parser.add_argument('--data_dir', type=str, 
                        default='./PestDataset/Combined_pestDataset',
                        help='Path to dataset directory')
    parser.add_argument('--save_dir', type=str, default='./models',
                        help='Directory to save models (default: ./models)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (default: auto-detect)')
    return parser.parse_args()

def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Create models directory if it doesn't exist
    Path("./models").mkdir(exist_ok=True)
    
    # Set device
    if args.device:
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Training configuration
    CONFIG = {
        'model_name': args.model,
        'num_epochs': args.epochs,
        'learning_rate': args.lr if not args.continue_training else 0.0001,
        'batch_size': args.batch_size,
        'input_size': 224,
        'device': device,
        'save_dir': './models'
    }
    
    print("🌾 Fall Armyworm Detection - Training Pipeline")
    print("=" * 60)
    
    # Load data from CSV files
    data_config = {
        'csv_path': './PestDataset/Combined_pestDataset/Train.csv',
        'images_dir': './PestDataset/Combined_pestDataset/Images',
        'batch_size': CONFIG['batch_size'],
        'input_size': CONFIG['input_size'],
        'train_split': 0.8,
        'num_workers': 4,
        'pin_memory': torch.cuda.is_available()
    }
    
    # Create data loaders
    print("📂 Loading dataset...")
    train_loader, val_loader, class_counts = create_data_loaders(**data_config)
    
    if args.continue_training:
        # Continue training from checkpoint
        trained_model, metrics, best_auc = continue_training(
            model_path='./models/best_model.pth',
            train_loader=train_loader,
            val_loader=val_loader,
            additional_epochs=CONFIG['num_epochs'],
            learning_rate=CONFIG['learning_rate'],
            device=CONFIG['device'],
            save_dir=CONFIG['save_dir']
        )
    else:
        # Initialize new model
        model = FallArmywormClassifier(
            model_name=CONFIG['model_name'],
            pretrained=True,
            dropout=0.3
        )
        
        print(f"🤖 Model initialized: {CONFIG['model_name']}")
        print(f"📊 Model size: {model.get_model_size():.2f} MB")
        
        # Train the model using our custom function
        trained_model, metrics, best_auc = custom_train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=CONFIG['num_epochs'],
            learning_rate=CONFIG['learning_rate'],
            device=CONFIG['device'],
            save_dir=CONFIG['save_dir'],
            disable_early_stopping=args.no_early_stopping
        )
    
    # Evaluate the model
    final_auc, accuracy, probs, labels = evaluate_model(
        trained_model, val_loader, CONFIG['device'], CONFIG['save_dir']
    )
    
    # Export for mobile deployment
    export_for_mobile(
        trained_model, 
        input_size=CONFIG['input_size'], 
        save_dir=CONFIG['save_dir']
    )
    
    print(f"\n🎉 Training complete!")
    print(f"🏆 Final AUC: {final_auc:.4f}")
    print(f"📱 Models exported for mobile deployment")
    print(f"💾 All files saved to: {CONFIG['save_dir']}")
    print("\nTo generate submission file, run: python generate_submission.py")

if __name__ == "__main__":
    main()