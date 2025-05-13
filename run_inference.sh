#!/bin/bash
# run_inference.sh - Script to run inference on calcification images

# Default paths
MODEL="models/retinanet_calc_resnet50.pt"
IMAGE_PATH=""
OUTPUT_DIR="inference_results"
SCORE_THRESHOLD=0.5
CLASS_MAP="ddsm_calc_retinanet_data/class_map.csv"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --model)
      MODEL="$2"
      shift 2
      ;;
    --image)
      IMAGE_PATH="$2"
      shift 2
      ;;
    --output)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --threshold)
      SCORE_THRESHOLD="$2"
      shift 2
      ;;
    --class_map)
      CLASS_MAP="$2"
      shift 2
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo
      echo "Options:"
      echo "  --model FILE         Path to trained model file"
      echo "  --image FILE/DIR     Path to input image or directory"
      echo "  --output DIR         Output directory for results"
      echo "  --threshold FLOAT    Detection threshold (default: 0.5)"
      echo "  --class_map FILE     Path to class map CSV file"
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

# Check if model file exists
if [ ! -f "$MODEL" ]; then
    echo "Error: Model file not found: $MODEL"
    exit 1
fi

# Check if class map file exists
if [ ! -f "$CLASS_MAP" ]; then
    echo "Error: Class map file not found: $CLASS_MAP"
    exit 1
fi

# Check if image path is provided
if [ -z "$IMAGE_PATH" ]; then
    echo "Error: No image path provided"
    echo "Use --image to specify an image or directory path"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run the inference script
echo "Starting inference..."
echo "Model: $MODEL"
echo "Image path: $IMAGE_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Detection threshold: $SCORE_THRESHOLD"
echo

$PYTHON_CMD inference.py --model "$MODEL" --image "$IMAGE_PATH" --output "$OUTPUT_DIR" --score-threshold "$SCORE_THRESHOLD" --class-list "$CLASS_MAP"

# Check if the script ran successfully
if [ $? -eq 0 ]; then
    echo
    echo "Inference completed successfully!"
    echo "Results saved to: $OUTPUT_DIR"
else
    echo
    echo "Error: Inference failed."
    exit 1
fi
