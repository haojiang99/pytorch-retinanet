#!/bin/bash
# train_calc_model.sh - Script to train RetinaNet on calcification data

# Default paths
ANNOTATIONS="ddsm_calc_retinanet_data/annotations.csv"
CLASS_MAP="ddsm_calc_retinanet_data/class_map.csv"
DEPTH=50
BATCH_SIZE=2
EPOCHS=50
LEARNING_RATE=1e-5
OUTPUT_DIR="models"
CHECKPOINT=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --annotations)
      ANNOTATIONS="$2"
      shift 2
      ;;
    --class_map)
      CLASS_MAP="$2"
      shift 2
      ;;
    --depth)
      DEPTH="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --lr)
      LEARNING_RATE="$2"
      shift 2
      ;;
    --output)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --checkpoint)
      CHECKPOINT="--model $2"
      shift 2
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo
      echo "Options:"
      echo "  --annotations FILE   Path to annotations CSV file"
      echo "  --class_map FILE     Path to class_map CSV file"
      echo "  --depth N            Depth of ResNet model (default: 50)"
      echo "  --batch-size N       Batch size (default: 2)"
      echo "  --epochs N           Number of epochs (default: 50)"
      echo "  --lr RATE            Learning rate (default: 1e-5)"
      echo "  --output DIR         Output directory for models"
      echo "  --checkpoint FILE    Resume training from checkpoint"
      echo "  --help               Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Check if python is available
PYTHON_CMD=""
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "Error: Python not found. Please install Python 3.x"
    exit 1
fi

# Check if the annotations file exists
if [ ! -f "$ANNOTATIONS" ]; then
    echo "Error: Annotations file not found: $ANNOTATIONS"
    echo "Run prepare_calc_data.sh first to generate the annotations."
    exit 1
fi

# Check if the class map file exists
if [ ! -f "$CLASS_MAP" ]; then
    echo "Error: Class map file not found: $CLASS_MAP"
    echo "Run prepare_calc_data.sh first to generate the class map."
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run the training script
echo "Starting RetinaNet training..."
echo "Annotations file: $ANNOTATIONS"
echo "Class map file: $CLASS_MAP"
echo "ResNet depth: $DEPTH"
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "Learning rate: $LEARNING_RATE"
echo "Output directory: $OUTPUT_DIR"
if [ -n "$CHECKPOINT" ]; then
    echo "Resuming from checkpoint"
fi
echo

CUDA_VISIBLE_DEVICES=0 $PYTHON_CMD train.py \
    --dataset csv \
    --csv_train "$ANNOTATIONS" \
    --csv_classes "$CLASS_MAP" \
    --csv_val "$ANNOTATIONS" \
    --depth "$DEPTH" \
    --epochs "$EPOCHS" \
    $CHECKPOINT

# Move the model to the output directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MODEL_NAME="retinanet_calc_resnet${DEPTH}_${TIMESTAMP}.pt"
if [ -f "model_final.pt" ]; then
    mv model_final.pt "$OUTPUT_DIR/$MODEL_NAME"
    echo "Model saved as: $OUTPUT_DIR/$MODEL_NAME"
fi

# Check if the script ran successfully
if [ $? -eq 0 ]; then
    echo
    echo "Training completed successfully!"
else
    echo
    echo "Error: Training failed."
    exit 1
fi
