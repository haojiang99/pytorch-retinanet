#!/usr/bin/env python3
"""
Enhanced Training Script for RetinaNet on DDSM Dataset

Improvements over original train_ddsm.py:
- Advanced optimizers (AdamW with weight decay)
- Cosine annealing with warm restarts
- Early stopping with patience
- Best model tracking based on lowest loss
- Gradient accumulation for larger effective batch sizes
- Mixed precision training for speed and memory efficiency
- Learning rate warmup
- Better logging and monitoring
- Automatic model checkpoint management
"""

import argparse
import collections
import os
import csv
import time
import json
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
from torch.cuda.amp import GradScaler, autocast

from retinanet import model
from retinanet.dataloader import CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, Normalizer
from torch.utils.data import DataLoader

print(f'PyTorch version: {torch.__version__}')
print('CUDA available: {}'.format(torch.cuda.is_available()))
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name()}')
    print(f'CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')


def fix_image_paths(csv_file, dataset_path):
    """Verify and fix image paths in annotations file"""
    print("Verifying image paths in annotations file...")
    
    # Read the CSV file
    rows = []
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            rows.append(row)
    
    # Check and fix paths
    fixed_count = 0
    for i, row in enumerate(rows):
        if len(row) >= 6:  # path,x1,y1,x2,y2,class_name
            img_path = row[0]
            
            # Standardize path format - always use forward slashes for CSV
            img_path = img_path.replace('\\', '/')
            
            # Check absolute path first
            full_path = os.path.join(dataset_path, img_path)
            if not os.path.exists(full_path.replace('/', os.path.sep)):
                # Try just the filename in images directory
                filename = os.path.basename(img_path)
                alt_path = os.path.join(dataset_path, 'images', filename)
                
                if os.path.exists(alt_path):
                    # Update to correct path format
                    rows[i][0] = 'images/' + filename
                    fixed_count += 1
                    print(f"Fixed path: {img_path} -> images/{filename}")
                else:
                    print(f"Warning: Could not find image for {img_path}")
            else:
                # Path exists but might need standardization
                rows[i][0] = img_path
    
    if fixed_count > 0 or True:  # Always write back to ensure consistent formatting
        # Write back the fixed CSV
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        print(f"Fixed {fixed_count} image paths in {csv_file}")
    else:
        print("All image paths appear to be correct")


class BestModelTracker:
    """Track and save the best model based on lowest loss"""
    
    def __init__(self, checkpoint_dir, patience=20):
        self.checkpoint_dir = checkpoint_dir
        self.best_loss = float('inf')
        self.best_epoch = -1
        self.patience = patience
        self.patience_counter = 0
        self.best_model_path = os.path.join(checkpoint_dir, 'best_model.pt')
        self.training_log = []
        
    def update(self, epoch, loss, model):
        """Update best model if current loss is lower"""
        self.training_log.append({
            'epoch': epoch,
            'loss': loss,
            'is_best': False,
            'timestamp': datetime.now().isoformat()
        })
        
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_epoch = epoch
            self.patience_counter = 0
            
            # Save the best model (overwrite previous)
            print(f"\n🎯 New best model found at epoch {epoch}!")
            print(f"   Previous best loss: {self.best_loss:.6f} -> New best loss: {loss:.6f}")
            print(f"   Saving best model to: {self.best_model_path}")
            
            # Use weights_only=False for PyTorch 2.6+ compatibility
            try:
                torch.save(model.module if hasattr(model, 'module') else model, 
                          self.best_model_path, weights_only=False)
            except TypeError:
                # Fallback for older PyTorch versions
                torch.save(model.module if hasattr(model, 'module') else model, 
                          self.best_model_path)
            
            self.training_log[-1]['is_best'] = True
            
            # Save training progress
            log_path = os.path.join(self.checkpoint_dir, 'training_log.json')
            with open(log_path, 'w') as f:
                json.dump({
                    'best_loss': self.best_loss,
                    'best_epoch': self.best_epoch,
                    'training_history': self.training_log
                }, f, indent=2)
            
            return True
        else:
            self.patience_counter += 1
            return False
    
    def should_stop(self):
        """Check if training should stop due to no improvement"""
        return self.patience_counter >= self.patience
    
    def get_info(self):
        """Get current best model information"""
        return {
            'best_loss': self.best_loss,
            'best_epoch': self.best_epoch,
            'patience_counter': self.patience_counter,
            'patience_limit': self.patience
        }


def warmup_lr_scheduler(optimizer, warmup_epochs, warmup_factor):
    """Learning rate warmup scheduler"""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            alpha = epoch / warmup_epochs
            return warmup_factor * (1 - alpha) + alpha
        return 1
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main(args=None):
    parser = argparse.ArgumentParser(description='Enhanced Training script for RetinaNet on DDSM dataset.')

    parser.add_argument('--dataset_path', help='Path to dataset directory', default='ddsm_train3')
    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=500)
    parser.add_argument('--batch_size', help='Batch size', type=int, default=2)
    parser.add_argument('--accumulate_grad_batches', help='Gradient accumulation steps', type=int, default=4)
    parser.add_argument('--workers', help='Number of workers', type=int, default=4)
    parser.add_argument('--lr', help='Initial learning rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', help='Weight decay', type=float, default=1e-4)
    parser.add_argument('--checkpoint_dir', help='Directory to save checkpoints', default='ddsm_checkpoints_v2')
    parser.add_argument('--patience', help='Early stopping patience (epochs)', type=int, default=50)
    parser.add_argument('--warmup_epochs', help='Learning rate warmup epochs', type=int, default=5)
    parser.add_argument('--save_every', help='Save checkpoint every N epochs', type=int, default=25)
    parser.add_argument('--mixed_precision', help='Use mixed precision training', action='store_true', default=True)
    parser.add_argument('--resume', help='Resume from checkpoint path', type=str, default=None)

    parser = parser.parse_args(args)

    # Create checkpoint directory if it doesn't exist
    if not os.path.exists(parser.checkpoint_dir):
        os.makedirs(parser.checkpoint_dir)

    # Setup CSV file paths
    csv_train = os.path.join(parser.dataset_path, 'annotations.csv')
    csv_classes = os.path.join(parser.dataset_path, 'class_map.csv')
    
    # Ensure the required files exist
    if not os.path.exists(csv_train):
        raise ValueError(f"Annotations file not found: {csv_train}")
    if not os.path.exists(csv_classes):
        raise ValueError(f"Class map file not found: {csv_classes}")

    # Print configuration
    print("\n" + "="*80)
    print("🚀 ENHANCED DDSM RETINANET TRAINING")
    print("="*80)
    print(f"📂 Dataset: {parser.dataset_path}")
    print(f"📊 Annotations: {csv_train}")
    print(f"🏷️  Classes: {csv_classes}")
    print(f"🏗️  Model: ResNet-{parser.depth}")
    print(f"📈 Epochs: {parser.epochs}")
    print(f"🎯 Batch size: {parser.batch_size}")
    print(f"📈 Gradient accumulation: {parser.accumulate_grad_batches}")
    print(f"⚡ Learning rate: {parser.lr}")
    print(f"⚖️  Weight decay: {parser.weight_decay}")
    print(f"🛑 Early stopping patience: {parser.patience}")
    print(f"🔥 Mixed precision: {parser.mixed_precision}")
    print(f"💾 Checkpoint dir: {parser.checkpoint_dir}")
    print("="*80)
    
    # Verify and fix image paths in annotations if needed
    fix_image_paths(csv_train, parser.dataset_path)

    # Create the data loaders
    dataset_train = CSVDataset(train_file=csv_train, class_list=csv_classes,
                              transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))

    sampler = AspectRatioBasedSampler(dataset_train, batch_size=parser.batch_size, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=parser.workers, collate_fn=collater, batch_sampler=sampler)

    print(f"📊 Dataset loaded: {len(dataset_train)} training images")
    print(f"🎯 Number of classes: {dataset_train.num_classes()}")

    # Create the model
    print(f"\n🏗️  Creating ResNet-{parser.depth} RetinaNet model...")
    if parser.depth == 18:
        retinanet = model.resnet18(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 34:
        retinanet = model.resnet34(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 50:
        retinanet = model.resnet50(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 101:
        retinanet = model.resnet101(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 152:
        retinanet = model.resnet152(num_classes=dataset_train.num_classes(), pretrained=True)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

    # Move model to GPU if available
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        retinanet = retinanet.cuda()
        retinanet = torch.nn.DataParallel(retinanet).cuda()
        print(f"🔥 Model moved to GPU with DataParallel")
    else:
        retinanet = torch.nn.DataParallel(retinanet)
        print("💻 Using CPU training")

    # Initialize mixed precision scaler
    scaler = GradScaler() if parser.mixed_precision and use_gpu else None
    
    # Setup optimizer with weight decay
    optimizer = optim.AdamW(retinanet.parameters(), 
                           lr=parser.lr, 
                           weight_decay=parser.weight_decay,
                           betas=(0.9, 0.999),
                           eps=1e-8)
    
    # Setup learning rate schedulers
    warmup_scheduler = warmup_lr_scheduler(optimizer, parser.warmup_epochs, 0.1)
    main_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-7
    )
    
    # Initialize best model tracker
    model_tracker = BestModelTracker(parser.checkpoint_dir, patience=parser.patience)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if parser.resume and os.path.exists(parser.resume):
        print(f"📂 Resuming from checkpoint: {parser.resume}")
        try:
            checkpoint = torch.load(parser.resume, map_location='cpu', weights_only=False)
            if hasattr(checkpoint, 'load_state_dict'):
                retinanet.module.load_state_dict(checkpoint.state_dict())
            else:
                retinanet.module.load_state_dict(checkpoint)
            print("✅ Model state loaded successfully")
        except Exception as e:
            print(f"⚠️ Could not load checkpoint: {e}")

    loss_hist = collections.deque(maxlen=500)
    
    print(f"\n🚀 Starting training for {parser.epochs} epochs...")
    print("="*80)
    
    start_time = time.time()
    
    for epoch_num in range(start_epoch, parser.epochs):
        retinanet.train()
        if hasattr(retinanet, 'module'):
            retinanet.module.freeze_bn()
        else:
            retinanet.freeze_bn()

        epoch_losses = []
        epoch_start_time = time.time()
        
        # Learning rate warmup for first few epochs
        if epoch_num < parser.warmup_epochs:
            warmup_scheduler.step()
        else:
            main_scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\n📈 Epoch {epoch_num + 1}/{parser.epochs} | LR: {current_lr:.2e}")
        print("-" * 60)
        
        for iter_num, data in enumerate(dataloader_train):
            try:
                # Forward pass with mixed precision
                if parser.mixed_precision and use_gpu:
                    with autocast():
                        if use_gpu:
                            classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot']])
                        else:
                            classification_loss, regression_loss = retinanet([data['img'].float(), data['annot']])
                        
                        classification_loss = classification_loss.mean()
                        regression_loss = regression_loss.mean()
                        loss = classification_loss + regression_loss
                        
                        # Scale loss for gradient accumulation
                        loss = loss / parser.accumulate_grad_batches
                else:
                    if use_gpu:
                        classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot']])
                    else:
                        classification_loss, regression_loss = retinanet([data['img'].float(), data['annot']])
                        
                    classification_loss = classification_loss.mean()
                    regression_loss = regression_loss.mean()
                    loss = classification_loss + regression_loss
                    
                    # Scale loss for gradient accumulation
                    loss = loss / parser.accumulate_grad_batches

                if bool(loss == 0):
                    continue

                # Backward pass
                if parser.mixed_precision and use_gpu:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                # Gradient accumulation
                if (iter_num + 1) % parser.accumulate_grad_batches == 0:
                    if parser.mixed_precision and use_gpu:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 1.0)
                        optimizer.step()
                    
                    optimizer.zero_grad()

                # Record loss (unscaled)
                actual_loss = loss.item() * parser.accumulate_grad_batches
                loss_hist.append(actual_loss)
                epoch_losses.append(actual_loss)

                # Print progress
                if iter_num % 10 == 0:
                    cls_loss = classification_loss.item() * parser.accumulate_grad_batches
                    reg_loss = regression_loss.item() * parser.accumulate_grad_batches
                    running_loss = np.mean(list(loss_hist)[-50:])  # Last 50 iterations
                    
                    print(f"  Iter {iter_num:4d} | Cls: {cls_loss:.5f} | Reg: {reg_loss:.5f} | "
                          f"Total: {actual_loss:.5f} | Running: {running_loss:.5f}")

                del classification_loss, regression_loss, loss
                
            except Exception as e:
                print(f"⚠️ Error in iteration {iter_num}: {e}")
                continue

        # Calculate epoch statistics
        epoch_loss = np.mean(epoch_losses) if epoch_losses else float('inf')
        epoch_time = time.time() - epoch_start_time
        
        print(f"\n📊 Epoch {epoch_num + 1} completed in {epoch_time:.1f}s")
        print(f"   Average epoch loss: {epoch_loss:.6f}")
        
        # Update best model tracker
        is_best = model_tracker.update(epoch_num + 1, epoch_loss, retinanet)
        
        # Print tracker info
        tracker_info = model_tracker.get_info()
        print(f"   Best loss so far: {tracker_info['best_loss']:.6f} (epoch {tracker_info['best_epoch']})")
        print(f"   Patience: {tracker_info['patience_counter']}/{tracker_info['patience_limit']}")
        
        # Save regular checkpoint
        if (epoch_num + 1) % parser.save_every == 0:
            checkpoint_path = os.path.join(parser.checkpoint_dir, f'checkpoint_epoch_{epoch_num + 1}.pt')
            try:
                torch.save(retinanet.module if hasattr(retinanet, 'module') else retinanet, 
                          checkpoint_path, weights_only=False)
                print(f"💾 Saved checkpoint: {checkpoint_path}")
            except TypeError:
                torch.save(retinanet.module if hasattr(retinanet, 'module') else retinanet, 
                          checkpoint_path)
                print(f"💾 Saved checkpoint: {checkpoint_path}")
        
        # Early stopping check
        if model_tracker.should_stop():
            print(f"\n🛑 Early stopping triggered after {epoch_num + 1} epochs")
            print(f"   No improvement for {parser.patience} epochs")
            break
    
    # Training completed
    total_time = time.time() - start_time
    print("\n" + "="*80)
    print("🎉 TRAINING COMPLETED!")
    print("="*80)
    print(f"⏱️  Total training time: {total_time / 3600:.2f} hours")
    print(f"🎯 Best model loss: {model_tracker.best_loss:.6f} (epoch {model_tracker.best_epoch})")
    print(f"💾 Best model saved at: {model_tracker.best_model_path}")
    
    # Save final model
    final_model_path = os.path.join(parser.checkpoint_dir, 'final_model.pt')
    try:
        torch.save(retinanet.module if hasattr(retinanet, 'module') else retinanet, 
                  final_model_path, weights_only=False)
    except TypeError:
        torch.save(retinanet.module if hasattr(retinanet, 'module') else retinanet, 
                  final_model_path)
    
    print(f"💾 Final model saved at: {final_model_path}")
    print("="*80)


if __name__ == '__main__':
    main()