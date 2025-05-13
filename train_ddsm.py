import argparse
import collections
import os

import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms

from retinanet import model
from retinanet.dataloader import CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, Normalizer
from torch.utils.data import DataLoader

from retinanet import csv_eval

print(f'PyTorch version: {torch.__version__}')
print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
    parser = argparse.ArgumentParser(description='Training script for RetinaNet on DDSM dataset.')

    parser.add_argument('--dataset_path', help='Path to dataset directory', default='ddsm_train_mini')
    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=100)
    parser.add_argument('--batch_size', help='Batch size', type=int, default=2)
    parser.add_argument('--workers', help='Number of workers', type=int, default=4)
    parser.add_argument('--lr', help='Learning rate', type=float, default=1e-5)
    parser.add_argument('--checkpoint_dir', help='Directory to save checkpoints', default='ddsm_checkpoints')

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

    # Print dataset information
    print(f"Training on DDSM dataset from {parser.dataset_path}")
    print(f"Annotations: {csv_train}")
    print(f"Class map: {csv_classes}")

    # Create the data loaders
    dataset_train = CSVDataset(train_file=csv_train, class_list=csv_classes,
                              transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))

    # No validation set for now
    dataset_val = None

    sampler = AspectRatioBasedSampler(dataset_train, batch_size=parser.batch_size, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=parser.workers, collate_fn=collater, batch_sampler=sampler)

    # Create the model
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
    else:
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.training = True

    optimizer = optim.Adam(retinanet.parameters(), lr=parser.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    loss_hist = collections.deque(maxlen=500)

    print('Num training images: {}'.format(len(dataset_train)))
    print(f'Training {parser.epochs} epochs with batch size {parser.batch_size}')
    print(f'Saving checkpoints every 10 epochs to {parser.checkpoint_dir}')

    for epoch_num in range(parser.epochs):
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

                if iter_num % 10 == 0:
                    print(
                        'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                            epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist)))

                del classification_loss
                del regression_loss
            except Exception as e:
                print(e)
                continue

        # Update learning rate
        scheduler.step(np.mean(epoch_loss))
        
        # Save model every 10 epochs or the final epoch
        if (epoch_num + 1) % 10 == 0 or epoch_num == parser.epochs - 1:
            torch.save(retinanet.module, os.path.join(parser.checkpoint_dir, f'ddsm_retinanet_{epoch_num}.pt'))
            print(f"Saved checkpoint at epoch {epoch_num}")

    # Save final model
    retinanet.eval()
    torch.save(retinanet, os.path.join(parser.checkpoint_dir, 'ddsm_retinanet_final.pt'))
    print("Training complete. Final model saved.")


if __name__ == '__main__':
    main()
