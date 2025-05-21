import argparse
import collections
import os
import csv
import sys
import time

import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms
from torch import serialization
from torch.nn.parallel import DataParallel

from retinanet import model
from retinanet.model import ResNet, Bottleneck, BasicBlock
from retinanet.dataloader import CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, Normalizer
from torch.utils.data import DataLoader

from retinanet import csv_eval

print(f'PyTorch version: {torch.__version__}')
print('CUDA available: {}'.format(torch.cuda.is_available()))


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


def main(args=None):
    parser = argparse.ArgumentParser(description='Continue training RetinaNet on DDSM dataset from a checkpoint.')

    parser.add_argument('--dataset_path', help='Path to dataset directory', default='ddsm_train3')
    parser.add_argument('--checkpoint', help='Path to checkpoint file to continue training from', required=True)
    parser.add_argument('--epochs', help='Number of additional epochs to train', type=int, default=400)
    parser.add_argument('--batch_size', help='Batch size', type=int, default=2)
    parser.add_argument('--workers', help='Number of workers', type=int, default=4)
    parser.add_argument('--lr', help='Learning rate', type=float, default=1e-5)
    parser.add_argument('--checkpoint_dir', help='Directory to save checkpoints', default='ddsm_checkpoints')
    parser.add_argument('--start_epoch', help='Starting epoch number (for naming)', type=int, default=100)
    parser.add_argument('--weights_only', help='Force loading with weights_only set to True', action='store_true')

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
    if not os.path.exists(parser.checkpoint):
        raise ValueError(f"Checkpoint file not found: {parser.checkpoint}")

    # Print dataset information
    print(f"Continuing training on DDSM dataset from {parser.dataset_path}")
    print(f"Using checkpoint: {parser.checkpoint}")
    print(f"Training for {parser.epochs} additional epochs")
    print(f"Annotations: {csv_train}")
    print(f"Class map: {csv_classes}")
    
    # Verify and fix image paths in annotations if needed
    fix_image_paths(csv_train, parser.dataset_path)

    # Create the data loaders
    dataset_train = CSVDataset(train_file=csv_train, class_list=csv_classes,
                              transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))

    # No validation set for now
    dataset_val = None

    sampler = AspectRatioBasedSampler(dataset_train, batch_size=parser.batch_size, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=parser.workers, collate_fn=collater, batch_sampler=sampler)

    # Load the model from checkpoint
    print(f"Loading model from checkpoint: {parser.checkpoint}")
    
    try:
        # Add safe globals for PyTorch 2.6+
        serialization.add_safe_globals([ResNet, Bottleneck, BasicBlock, DataParallel])
        
        # Try loading the model based on user preference for weights_only
        if parser.weights_only:
            print("Attempting to load model with weights_only=True as specified...")
            retinanet = torch.load(parser.checkpoint, map_location='cpu', weights_only=True)
        else:
            print("Attempting to load model with weights_only=False...")
            retinanet = torch.load(parser.checkpoint, map_location='cpu', weights_only=False)
        print("Loaded full model from checkpoint")
        
        # If it's wrapped in DataParallel, get the module
        if isinstance(retinanet, torch.nn.DataParallel):
            print("Model was saved with DataParallel, extracting module...")
            retinanet = retinanet.module
            
    except Exception as e:
        print(f"First loading attempt failed: {e}")
        
        try:
            # Try creating a new model and loading state dict
            print("Trying to create new model and load state dict...")
            
            # First, check how many classes by reading the class_list file
            class_count = 0
            with open(csv_classes, 'r') as f:
                reader = csv.reader(f)
                for _ in reader:
                    class_count += 1
            
            print(f"Creating new model with {class_count} classes based on class list")
            
            # Create a new model with the correct number of classes
            if parser.depth == 18:
                retinanet = model.resnet18(num_classes=class_count, pretrained=False)
            elif parser.depth == 34:
                retinanet = model.resnet34(num_classes=class_count, pretrained=False)
            elif parser.depth == 50:
                retinanet = model.resnet50(num_classes=class_count, pretrained=False)
            elif parser.depth == 101:
                retinanet = model.resnet101(num_classes=class_count, pretrained=False)
            elif parser.depth == 152:
                retinanet = model.resnet152(num_classes=class_count, pretrained=False)
            else:
                # Default to resnet50
                print(f"Using default ResNet-50 backbone")
                retinanet = model.resnet50(num_classes=class_count, pretrained=False)
            
            # Load state dict
            # Decide whether to load with weights_only based on user preference
            if parser.weights_only:
                print("Loading checkpoint with weights_only=True as specified...")
                state_dict = torch.load(parser.checkpoint, map_location='cpu', weights_only=True)
            else:
                try:
                    # Try first without weights_only
                    print("Loading checkpoint with weights_only=False...")
                    state_dict = torch.load(parser.checkpoint, map_location='cpu', weights_only=False)
                except Exception as e3:
                    print(f"Failed with weights_only=False, trying with weights_only=True: {e3}")
                    state_dict = torch.load(parser.checkpoint, map_location='cpu', weights_only=True)
            
            # If the loaded object is a full model, extract its state dict
            if isinstance(state_dict, torch.nn.Module):
                print("Loaded a full model, extracting state dict...")
                state_dict = state_dict.state_dict()
                
            # Load state dict into model
            retinanet.load_state_dict(state_dict)
            print("Successfully loaded state dict into new model")
            
        except Exception as e2:
            print(f"Error loading checkpoint: {e2}")
            print("Failed to load the model. Please try using an older PyTorch version or update the checkpoint.")
            sys.exit(1)

    # Verify model is properly loaded
    print(f"Model loaded successfully. Num classes: {dataset_train.num_classes()}")
    print(f"Moving model to {'GPU' if torch.cuda.is_available() else 'CPU'}")

    # Move model to GPU if available
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        retinanet = retinanet.cuda()
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.training = True

    # Setup optimizer with possibly reduced learning rate for fine-tuning
    optimizer = optim.Adam(retinanet.parameters(), lr=parser.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)

    loss_hist = collections.deque(maxlen=500)

    print('Num training images: {}'.format(len(dataset_train)))
    print(f"Starting from epoch {parser.start_epoch}, training {parser.epochs} more epochs")
    
    # Create a timestamp for this training run
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_file = os.path.join(parser.checkpoint_dir, f'training_log_{timestamp}.txt')
    
    # Log training parameters
    with open(log_file, 'w') as f:
        f.write(f"Training parameters:\n")
        f.write(f"Checkpoint: {parser.checkpoint}\n")
        f.write(f"Dataset: {parser.dataset_path}\n")
        f.write(f"Starting epoch: {parser.start_epoch}\n")
        f.write(f"Total epochs: {parser.start_epoch + parser.epochs}\n")
        f.write(f"Batch size: {parser.batch_size}\n")
        f.write(f"Learning rate: {parser.lr}\n")
        f.write(f"Workers: {parser.workers}\n")
        f.write(f"Checkpoint directory: {parser.checkpoint_dir}\n")
        f.write(f"PyTorch version: {torch.__version__}\n")
        f.write(f"CUDA available: {torch.cuda.is_available()}\n\n")
        f.write("Epoch,Iteration,Classification_Loss,Regression_Loss,Total_Loss\n")

    for epoch_num in range(parser.epochs):
        # Calculate actual epoch number for saving checkpoints
        actual_epoch = parser.start_epoch + epoch_num
        
        retinanet.train()
        retinanet.module.freeze_bn()

        epoch_loss = []

        for iter_num, data in enumerate(dataloader_train):
            try:
                optimizer.zero_grad()

                if use_gpu:
                    classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot']])
                else:
                    classification_loss, regression_loss = retinanet([data['img'].float(), data['annot']])
                    
                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()

                loss = classification_loss + regression_loss

                if bool(loss == 0):
                    continue

                loss.backward()

                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

                optimizer.step()

                loss_hist.append(float(loss))

                epoch_loss.append(float(loss))

                # Log losses
                log_message = f'Epoch: {actual_epoch} | Iter: {iter_num} | Class loss: {float(classification_loss):1.5f} | Reg loss: {float(regression_loss):1.5f} | Running loss: {np.mean(loss_hist):1.5f}'
                
                if iter_num % 10 == 0:
                    print(log_message)
                    
                    # Log to file
                    with open(log_file, 'a') as f:
                        f.write(f"{actual_epoch},{iter_num},{float(classification_loss)},{float(regression_loss)},{float(loss)}\n")

                del classification_loss
                del regression_loss
            except Exception as e:
                print(f"Error in iteration {iter_num}: {e}")
                continue

        # Update learning rate
        epoch_loss_mean = np.mean(epoch_loss)
        scheduler.step(epoch_loss_mean)
        
        print(f"Epoch {actual_epoch} complete. Average loss: {epoch_loss_mean:1.5f}")
        
        # Save model every 10 epochs or the final epoch
        if (epoch_num + 1) % 10 == 0 or epoch_num == parser.epochs - 1:
            checkpoint_path = os.path.join(parser.checkpoint_dir, f'ddsm_retinanet_{actual_epoch}.pt')
            torch.save(retinanet.module, checkpoint_path)
            print(f"Saved checkpoint at epoch {actual_epoch} to {checkpoint_path}")

    # Save final model with timestamp to avoid overwriting
    final_model_path = os.path.join(parser.checkpoint_dir, f'ddsm_retinanet_final_{timestamp}.pt')
    retinanet.eval()
    torch.save(retinanet.module, final_model_path)
    print(f"Extended training complete. Final model saved to {final_model_path}")
    
    # Also save with a standard name for easy reference
    standard_final_path = os.path.join(parser.checkpoint_dir, 'ddsm_retinanet_final.pt')
    torch.save(retinanet.module, standard_final_path)
    print(f"Also saved final model to standard path: {standard_final_path}")


if __name__ == '__main__':
    main()