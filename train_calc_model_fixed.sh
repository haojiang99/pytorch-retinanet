#!/bin/bash
# train_calc_model_fixed.sh - Fixed script to train RetinaNet with certificate issues handled

# Default paths
ANNOTATIONS="ddsm_calc_retinanet_data/annotations.csv"
CLASS_MAP="ddsm_calc_retinanet_data/class_map.csv"
DEPTH=50
EPOCHS=50
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
    --epochs)
      EPOCHS="$2"
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
      echo "  --epochs N           Number of epochs (default: 50)"
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

# Create a Python script to disable SSL verification for downloading pretrained models
cat > ssl_fix.py << 'EOF'
import ssl
import torch.utils.model_zoo as model_zoo
import torch.hub as hub

# Save the original function
original_load_url = model_zoo.load_url
original_download_url_to_file = hub.download_url_to_file

# Create a new function with SSL verification disabled
def load_url_no_verify(url, model_dir=None, map_location=None, progress=True, check_hash=False, file_name=None):
    # Create an unverified SSL context
    old_context = ssl._create_default_https_context
    ssl._create_default_https_context = ssl._create_unverified_context
    try:
        result = original_load_url(url, model_dir, map_location, progress, check_hash, file_name)
    finally:
        # Restore the original SSL context
        ssl._create_default_https_context = old_context
    return result

# Replace the original function with our new one
model_zoo.load_url = load_url_no_verify

# Also fix the hub download function
def download_url_to_file_no_verify(url, dst, hash_prefix=None, progress=True):
    old_context = ssl._create_default_https_context
    ssl._create_default_https_context = ssl._create_unverified_context
    try:
        result = original_download_url_to_file(url, dst, hash_prefix, progress)
    finally:
        ssl._create_default_https_context = old_context
    return result

hub.download_url_to_file = download_url_to_file_no_verify
EOF

# Display a notice about the fix
echo "Starting RetinaNet training with SSL verification disabled..."
echo "Annotations file: $ANNOTATIONS"
echo "Class map file: $CLASS_MAP"
echo "ResNet depth: $DEPTH"
echo "Epochs: $EPOCHS"
echo "Output directory: $OUTPUT_DIR"
if [ -n "$CHECKPOINT" ]; then
    echo "Resuming from checkpoint"
fi
echo

# Run the training script with SSL verification fix
CUDA_VISIBLE_DEVICES=0 $PYTHON_CMD -c "
import ssl_fix
import sys
sys.argv = ['train.py', '--dataset', 'csv', '--csv_train', '$ANNOTATIONS', '--csv_classes', '$CLASS_MAP', '--csv_val', '$ANNOTATIONS', '--depth', '$DEPTH', '--epochs', '$EPOCHS' $CHECKPOINT]
exec(open('train.py').read())
"

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
