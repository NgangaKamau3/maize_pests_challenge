import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import seaborn as sns
from pathlib import Path
import time
import copy
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class FallArmywormClassifier(nn.Module):
    """Mobile-optimized binary classifier for fall armyworm detection"""
    
    def __init__(self, model_name='mobilenet_v2', pretrained=True, dropout=0.3):
        super(FallArmywormClassifier, self).__init__()
        self.model_name = model_name
        
        if model_name == 'mobilenet_v2':
            # MobileNetV2 - Optimal for mobile deployment
            self.backbone = models.mobilenet_v2(pretrained=pretrained)
            # Replace classifier head
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(self.backbone.last_channel, 1)  # Binary classification
            )
        elif model_name == 'resnet18':
            # ResNet18 - Alternative lightweight option
            self.backbone = models.resnet18(pretrained=pretrained)
            # Replace final layer
            self.backbone.fc = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(self.backbone.fc.in_features, 1)
            )
        else:
            raise ValueError(f"Unsupported model: {model_name}")
    
    def forward(self, x):
        return self.backbone(x)
    
    def get_model_size(self):
        """Calculate model size in MB"""
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        return (param_size + buffer_size) / (1024**2)

class EarlyStopping:
    """Early stopping based on validation AUC"""
    
    def __init__(self, patience=7, min_delta=0.001, restore_best=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.best_score = None
        self.counter = 0
        self.best_weights = None
    
    def __call__(self, val_auc, model):
        if self.best_score is None:
            self.best_score = val_auc
            self.save_checkpoint(model)
        elif val_auc > self.best_score + self.min_delta:
            self.best_score = val_auc
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
        
        return self.counter >= self.patience
    
    def save_checkpoint(self, model):
        if self.restore_best:
            self.best_weights = copy.deepcopy(model.state_dict())
    
    def restore(self, model):
        if self.restore_best and self.best_weights:
            model.load_state_dict(self.best_weights)

class MetricsTracker:
    """Track and visualize training metrics"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.train_losses = []
        self.val_losses = []
        self.train_aucs = []
        self.val_aucs = []
        self.learning_rates = []
    
    def update(self, train_loss, val_loss, train_auc, val_auc, lr):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_aucs.append(train_auc)
        self.val_aucs.append(val_auc)
        self.learning_rates.append(lr)
    
    def plot_metrics(self, save_path=None):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Metrics - Fall Armyworm Detection', fontsize=16)
        
        # Loss curves
        axes[0, 0].plot(self.train_losses, label='Train Loss', color='blue')
        axes[0, 0].plot(self.val_losses, label='Val Loss', color='red')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('BCE Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # AUC curves
        axes[0, 1].plot(self.train_aucs, label='Train AUC', color='blue')
        axes[0, 1].plot(self.val_aucs, label='Val AUC', color='red')
        axes[0, 1].set_title('AUC Curves')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('AUC Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning rate
        axes[1, 0].plot(self.learning_rates, color='green')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].grid(True)
        
        # Best metrics summary
        best_val_auc = max(self.val_aucs)
        best_epoch = self.val_aucs.index(best_val_auc)
        axes[1, 1].text(0.1, 0.8, f'Best Val AUC: {best_val_auc:.4f}', fontsize=12)
        axes[1, 1].text(0.1, 0.6, f'Best Epoch: {best_epoch + 1}', fontsize=12)
        axes[1, 1].text(0.1, 0.4, f'Final Train AUC: {self.train_aucs[-1]:.4f}', fontsize=12)
        axes[1, 1].text(0.1, 0.2, f'Final Val AUC: {self.val_aucs[-1]:.4f}', fontsize=12)
        axes[1, 1].set_title('Training Summary')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

def calculate_auc(y_true, y_pred):
    """Calculate AUC score safely"""
    try:
        if len(np.unique(y_true)) < 2:
            return 0.5  # Random performance if only one class
        return roc_auc_score(y_true, y_pred)
    except:
        return 0.5

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
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
    
    epoch_loss = running_loss / len(train_loader)
    epoch_auc = calculate_auc(all_labels, all_preds)
    
    return epoch_loss, epoch_auc

def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch"""
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
    
    epoch_loss = running_loss / len(val_loader)
    epoch_auc = calculate_auc(all_labels, all_preds)
    
    return epoch_loss, epoch_auc, all_preds, all_labels

def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001, 
                device='cuda', save_dir='./models'):
    """Complete training pipeline with early stopping and metrics tracking"""
    
    # Setup
    Path(save_dir).mkdir(exist_ok=True)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, 
                                                     factor=0.5, verbose=True)
    
    # Early stopping and metrics tracking
    early_stopping = EarlyStopping(patience=10, min_delta=0.001)
    metrics = MetricsTracker()
    
    # Move model to device
    model = model.to(device)
    
    print(f"🚀 Starting training on {device}")
    print(f"📱 Model size: {model.get_model_size():.2f} MB")
    print(f"🎯 Target: Mobile-optimized fall armyworm detection")
    print("-" * 60)
    
    best_val_auc = 0.0
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Training phase
        train_loss, train_auc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validation phase
        val_loss, val_auc, val_preds, val_labels = validate_epoch(model, val_loader, criterion, device)
        
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
        if early_stopping(val_auc, model):
            print(f'🛑 Early stopping triggered after {epoch+1} epochs')
            break
    
    # Restore best weights
    early_stopping.restore(model)
    
    training_time = time.time() - start_time
    print(f"\n✅ Training completed!")
    print(f"⏱️  Total time: {training_time/60:.1f} minutes")
    print(f"🏆 Best validation AUC: {best_val_auc:.4f}")
    
    # Plot training metrics
    metrics.plot_metrics(save_path=f'{save_dir}/training_metrics.png')
    
    return model, metrics, best_val_auc

def evaluate_model(model, val_loader, device, save_dir='./models'):
    """Comprehensive model evaluation"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for data, target in tqdm(val_loader, desc='Evaluating'):
            data = data.to(device)
            outputs = model(data).squeeze()
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(target.numpy())
    
    # Calculate metrics
    auc = calculate_auc(all_labels, all_probs)
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    cm = confusion_matrix(all_labels, all_preds)
    
    print(f"\n📊 Model Evaluation Results:")
    print(f"🎯 AUC Score: {auc:.4f}")
    print(f"✅ Accuracy: {accuracy:.4f}")
    print(f"📱 Model Size: {model.get_model_size():.2f} MB")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Healthy', 'Infected'],
                yticklabels=['Healthy', 'Infected'])
    plt.title('Confusion Matrix - Fall Armyworm Detection')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Classification report
    print(f"\n📋 Detailed Classification Report:")
    print(classification_report(all_labels, all_preds, 
                              target_names=['Healthy', 'Infected']))
    
    return auc, accuracy, all_probs, all_labels

def export_for_mobile(model, input_size=224, save_dir='./models'):
    """Export model for mobile deployment"""
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, input_size, input_size)
    
    # Export to TorchScript (mobile-friendly)
    try:
        traced_model = torch.jit.trace(model, dummy_input)
        torch.jit.save(traced_model, f'{save_dir}/mobile_model.pt')
        print(f"📱 TorchScript model saved: {save_dir}/mobile_model.pt")
    except Exception as e:
        print(f"❌ TorchScript export failed: {e}")
    
    # Export to ONNX (cross-platform)
    try:
        torch.onnx.export(model, dummy_input, f'{save_dir}/mobile_model.onnx',
                         export_params=True, opset_version=11,
                         input_names=['input'], output_names=['output'],
                         dynamic_axes={'input': {0: 'batch_size'},
                                     'output': {0: 'batch_size'}})
        print(f"🌐 ONNX model saved: {save_dir}/mobile_model.onnx")
    except Exception as e:
        print(f"❌ ONNX export failed: {e}")

# Example usage and training pipeline
if __name__ == "__main__":
    # Training configuration optimized for mobile deployment
    CONFIG = {
        'model_name': 'mobilenet_v2',  # Optimal for mobile
        'num_epochs': 50,
        'learning_rate': 0.001,
        'batch_size': 32,
        'input_size': 224,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_dir': './models'
    }
    
    print("🌾 Fall Armyworm Detection System - Mobile Training Pipeline")
    print("="*60)
    
    # Assuming you have data loaders from the previous script
    # Replace this with your actual data loading code
    """
    from data_preprocessing import create_data_loaders
    train_loader, val_loader, class_counts = create_data_loaders(
        data_dir='path/to/your/dataset',
        batch_size=CONFIG['batch_size'],
        input_size=CONFIG['input_size']
    )
    """
    
    # For demonstration, let's create dummy loaders
    print("⚠️  Using dummy data loaders for demonstration")
    print("   Replace with your actual data loaders from the preprocessing script")
    
    # Initialize model
    model = FallArmywormClassifier(
        model_name=CONFIG['model_name'],
        pretrained=True,
        dropout=0.3
    )
    
    print(f"🤖 Model initialized: {CONFIG['model_name']}")
    print(f"📊 Model size: {model.get_model_size():.2f} MB")
    
    # Uncomment when you have real data loaders:
    """
    # Train the model
    trained_model, metrics, best_auc = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=CONFIG['num_epochs'],
        learning_rate=CONFIG['learning_rate'],
        device=CONFIG['device'],
        save_dir=CONFIG['save_dir']
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
    """
    
    print("\n🚀 Ready to train! Update the data loader section and run the training.")