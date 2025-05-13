#!/bin/bash
# prepare_calc_data.sh - Script to prepare calcification data from DDSM for RetinaNet

# Default paths
CSV_FILE="/Users/haojiang/Documents/DDSM/manifest-ZkhPvrLo5216730872708713142/calc_case_description_test_set.csv"
DDSM_DIR="/Users/haojiang/Documents/DDSM/manifest-ZkhPvrLo5216730872708713142/CBIS-DDSM"
OUTPUT_DIR="ddsm_calc_retinanet_data"
LIMIT=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --csv)
      CSV_FILE="$2"
      shift 2
      ;;
    --ddsm)
      DDSM_DIR="$2"
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
      echo "Usage: $0 [--csv CSV_FILE] [--ddsm DDSM_DIR] [--output OUTPUT_DIR] [--limit N]"
      echo
      echo "Options:"
      echo "  --csv FILE      Path to calc_case_description_test_set.csv"
      echo "  --ddsm DIR      Path to CBIS-DDSM directory"
      echo "  --output DIR    Output directory for prepared data"
      echo "  --limit N       Limit processing to N samples"
      echo "  --help          Show this help message"
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

# Check if the CSV file exists
if [ ! -f "$CSV_FILE" ]; then
    echo "Error: CSV file not found: $CSV_FILE"
    exit 1
fi

# Check if the DDSM directory exists
if [ ! -d "$DDSM_DIR" ]; then
    echo "Error: DDSM directory not found: $DDSM_DIR"
    exit 1
fi

# Run the prepare_ddsm_data.py script
echo "Starting data preparation..."
echo "CSV file: $CSV_FILE"
echo "DDSM directory: $DDSM_DIR"
echo "Output directory: $OUTPUT_DIR"
if [ -n "$LIMIT" ]; then
    echo "Processing limit: $LIMIT samples"
fi
echo

$PYTHON_CMD prepare_ddsm_data.py --csv_file "$CSV_FILE" --ddsm_dir "$DDSM_DIR" --output_dir "$OUTPUT_DIR" $LIMIT

# Check if the script ran successfully
if [ $? -eq 0 ]; then
    echo
    echo "Data preparation completed successfully!"
else
    echo
    echo "Error: Data preparation failed."
    exit 1
fi
