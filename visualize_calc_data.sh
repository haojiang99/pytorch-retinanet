#!/bin/bash
# visualize_calc_data.sh - Script to visualize prepared calcification data

# Default paths
ANNOTATIONS="ddsm_calc_retinanet_data/annotations.csv"
CLASS_MAP="ddsm_calc_retinanet_data/class_map.csv"
OUTPUT_DIR="ddsm_calc_visualization"
LIMIT=""

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
    --output)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --limit)
      LIMIT="--limit $2"
      shift 2
      ;;
    --help)
      echo "Usage: $0 [--annotations FILE] [--class_map FILE] [--output DIR] [--limit N]"
      echo
      echo "Options:"
      echo "  --annotations FILE  Path to annotations CSV file"
      echo "  --class_map FILE    Path to class_map CSV file"
      echo "  --output DIR        Output directory for visualization images"
      echo "  --limit N           Limit visualization to N samples"
      echo "  --help              Show this help message"
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

# Run the visualization script
echo "Starting visualization..."
echo "Annotations file: $ANNOTATIONS"
echo "Class map file: $CLASS_MAP"
echo "Output directory: $OUTPUT_DIR"
if [ -n "$LIMIT" ]; then
    echo "Processing limit: $LIMIT samples"
fi
echo

$PYTHON_CMD visualize_calc_data.py --annotations "$ANNOTATIONS" --class_map "$CLASS_MAP" --output_dir "$OUTPUT_DIR" $LIMIT

# Check if the script ran successfully
if [ $? -eq 0 ]; then
    echo
    echo "Visualization completed successfully!"
    echo "Check the output directory: $OUTPUT_DIR"
else
    echo
    echo "Error: Visualization failed."
    exit 1
fi
