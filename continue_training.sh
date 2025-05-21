#!/bin/bash

# continue_training.sh - Easy script to continue training RetinaNet on DDSM dataset
# 
# This script provides a simple way to continue training from a checkpoint
# with sensible defaults while allowing customization of key parameters.

# Default values
DATASET_PATH="ddsm_train3"
CHECKPOINT_DIR="ddsm_checkpoints"
DEFAULT_CHECKPOINT="${CHECKPOINT_DIR}/ddsm_retinanet_final.pt"
ADDITIONAL_EPOCHS=400
BATCH_SIZE=2
LEARNING_RATE=5e-6
NUM_WORKERS=4
START_EPOCH=100
WEIGHTS_ONLY=false

# Display help message
show_help() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Continue training RetinaNet on DDSM dataset from a checkpoint."
    echo ""
    echo "Options:"
    echo "  -c, --checkpoint FILE    Checkpoint file to continue from (default: ${DEFAULT_CHECKPOINT})"
    echo "  -d, --dataset PATH       Path to dataset directory (default: ${DATASET_PATH})"
    echo "  -e, --epochs NUMBER      Number of additional epochs (default: ${ADDITIONAL_EPOCHS})"
    echo "  -b, --batch-size NUMBER  Batch size (default: ${BATCH_SIZE})"
    echo "  -l, --lr NUMBER          Learning rate (default: ${LEARNING_RATE})"
    echo "  -w, --workers NUMBER     Number of data loader workers (default: ${NUM_WORKERS})"
    echo "  -s, --start-epoch NUMBER Starting epoch number (default: ${START_EPOCH})"
    echo "  --weights-only           Force loading checkpoint with weights_only=True (for PyTorch 2.6+)"
    echo "  -h, --help               Show this help message and exit"
    echo ""
    echo "Example:"
    echo "  $0 --checkpoint ${CHECKPOINT_DIR}/ddsm_retinanet_90.pt --epochs 200 --start-epoch 90"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    
    case $key in
        -c|--checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        -d|--dataset)
            DATASET_PATH="$2"
            shift 2
            ;;
        -e|--epochs)
            ADDITIONAL_EPOCHS="$2"
            shift 2
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -l|--lr)
            LEARNING_RATE="$2"
            shift 2
            ;;
        -w|--workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        -s|--start-epoch)
            START_EPOCH="$2"
            shift 2
            ;;
        --weights-only)
            WEIGHTS_ONLY=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Use default checkpoint if not specified
if [ -z "$CHECKPOINT" ]; then
    CHECKPOINT=$DEFAULT_CHECKPOINT
    echo "No checkpoint specified, using default: $CHECKPOINT"
fi

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint file not found: $CHECKPOINT"
    exit 1
fi

# Check if dataset exists
if [ ! -d "$DATASET_PATH" ]; then
    echo "Error: Dataset directory not found: $DATASET_PATH"
    exit 1
fi

# Check if checkpoint directory exists
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "Creating checkpoint directory: $CHECKPOINT_DIR"
    mkdir -p "$CHECKPOINT_DIR"
fi

# Display training details
echo "=========================================================="
echo "                CONTINUE TRAINING RETINANET               "
echo "=========================================================="
echo "Dataset path:     $DATASET_PATH"
echo "Checkpoint file:  $CHECKPOINT"
echo "Additional epochs: $ADDITIONAL_EPOCHS"
echo "Starting from epoch: $START_EPOCH"
echo "Batch size:       $BATCH_SIZE"
echo "Learning rate:    $LEARNING_RATE"
echo "Workers:          $NUM_WORKERS"
echo "Checkpoint dir:   $CHECKPOINT_DIR"
echo "Weights only:     $WEIGHTS_ONLY"
echo "=========================================================="
echo "Starting training in 3 seconds... Press Ctrl+C to cancel"
echo "=========================================================="

# Wait for 3 seconds to allow user to cancel if needed
sleep 3

# Start the Python script with the specified parameters
if [ "$WEIGHTS_ONLY" = true ]; then
    python continue_training_ddsm.py \
        --checkpoint "$CHECKPOINT" \
        --dataset_path "$DATASET_PATH" \
        --epochs "$ADDITIONAL_EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --lr "$LEARNING_RATE" \
        --workers "$NUM_WORKERS" \
        --start_epoch "$START_EPOCH" \
        --checkpoint_dir "$CHECKPOINT_DIR" \
        --weights_only
else
    python continue_training_ddsm.py \
        --checkpoint "$CHECKPOINT" \
        --dataset_path "$DATASET_PATH" \
        --epochs "$ADDITIONAL_EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --lr "$LEARNING_RATE" \
        --workers "$NUM_WORKERS" \
        --start_epoch "$START_EPOCH" \
        --checkpoint_dir "$CHECKPOINT_DIR"
fi

# Check exit status
if [ $? -eq 0 ]; then
    echo "=========================================================="
    echo "Training completed successfully!"
    echo "Final model saved to: ${CHECKPOINT_DIR}/ddsm_retinanet_final.pt"
    echo "=========================================================="
else
    echo "=========================================================="
    echo "Training failed with an error. Please check the logs."
    echo "=========================================================="
    exit 1
fi