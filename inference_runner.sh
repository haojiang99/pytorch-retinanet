#!/bin/bash
# inference_runner.sh - Script to run pytorch-retinanet inference on a single image

# Print usage instructions
usage() {
    echo "Usage: $0 <image_path> [output_path]"
    echo
    echo "Arguments:"
    echo "  <image_path>   Path to the input image for detection"
    echo "  [output_path]  Optional path for saving the output image (default: output.jpg)"
    echo
    echo "Example:"
    echo "  $0 test.jpeg detection_output.jpg"
    exit 1
}

# Check if image path is provided
if [ -z "$1" ]; then
    echo "Error: Image path is required"
    usage
fi

# Set variables
IMAGE_PATH="$1"
OUTPUT_PATH="${2:-output.jpg}"
MODEL_PATH="coco_resnet_50_map_0_335_state_dict.pt"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if the image exists
if [ ! -f "$IMAGE_PATH" ]; then
    echo "Error: Image file '$IMAGE_PATH' does not exist"
    exit 1
fi

# Check if the model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file '$MODEL_PATH' does not exist"
    exit 1
fi

# Instructions for running the script
echo "==========================================================="
echo "PyTorch RetinaNet Inference"
echo "==========================================================="
echo "Image path: $IMAGE_PATH"
echo "Output will be saved to: $OUTPUT_PATH"
echo "Model: $MODEL_PATH"
echo
echo "Before running this script, you need to set up your environment:"
echo
echo "Option 1: If you have conda/miniconda installed:"
echo "  conda create -n retinanet python=3.8"
echo "  conda activate retinanet"
echo "  pip install torch torchvision numpy opencv-python"
echo
echo "Option 2: If you prefer using a virtual environment:"
echo "  python -m venv venv"
echo "  source venv/bin/activate  # On Windows, use: venv\\Scripts\\activate"
echo "  pip install torch torchvision numpy opencv-python"
echo
echo "Once your environment is set up, run:"
echo "  python $SCRIPT_DIR/inference.py --image $IMAGE_PATH --output $OUTPUT_PATH --model $MODEL_PATH"
echo
echo "==========================================================="
