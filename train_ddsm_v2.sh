#!/bin/bash

# Enhanced DDSM Training Script v2
# This script runs the improved training with optimized hyperparameters

echo "🚀 Starting Enhanced DDSM RetinaNet Training v2"
echo "=============================================="

# Set training parameters
DATASET_PATH="ddsm_train3"
EPOCHS=500
BATCH_SIZE=2
ACCUMULATE_GRAD=4  # Effective batch size = 2 * 4 = 8
LEARNING_RATE=1e-4
WEIGHT_DECAY=1e-4
DEPTH=50
PATIENCE=50
WARMUP_EPOCHS=5
SAVE_EVERY=25
CHECKPOINT_DIR="ddsm_checkpoints_v2"

# Check if dataset exists
if [ ! -d "$DATASET_PATH" ]; then
    echo "❌ Error: Dataset directory '$DATASET_PATH' not found!"
    echo "Please ensure the dataset is available or update DATASET_PATH"
    exit 1
fi

# Check for required files
if [ ! -f "$DATASET_PATH/annotations.csv" ]; then
    echo "❌ Error: annotations.csv not found in $DATASET_PATH"
    exit 1
fi

if [ ! -f "$DATASET_PATH/class_map.csv" ]; then
    echo "❌ Error: class_map.csv not found in $DATASET_PATH"
    exit 1
fi

# Create checkpoint directory
mkdir -p "$CHECKPOINT_DIR"

echo "📋 Training Configuration:"
echo "   Dataset: $DATASET_PATH"
echo "   Epochs: $EPOCHS"
echo "   Batch Size: $BATCH_SIZE (effective: $((BATCH_SIZE * ACCUMULATE_GRAD)))"
echo "   Learning Rate: $LEARNING_RATE"
echo "   Weight Decay: $WEIGHT_DECAY"
echo "   Model: ResNet-$DEPTH"
echo "   Early Stopping Patience: $PATIENCE epochs"
echo "   Checkpoints: $CHECKPOINT_DIR"
echo ""

# Start training
python train_ddsm_v2.py \
    --dataset_path "$DATASET_PATH" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --accumulate_grad_batches "$ACCUMULATE_GRAD" \
    --lr "$LEARNING_RATE" \
    --weight_decay "$WEIGHT_DECAY" \
    --depth "$DEPTH" \
    --patience "$PATIENCE" \
    --warmup_epochs "$WARMUP_EPOCHS" \
    --save_every "$SAVE_EVERY" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --mixed_precision \
    --workers 4

echo ""
echo "✅ Training script completed!"
echo "Check the results in: $CHECKPOINT_DIR"