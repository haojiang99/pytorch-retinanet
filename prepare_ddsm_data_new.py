#!/usr/bin/env python3
"""
Modified DDSM Data Preparation Script for RetinaNet Training

This script processes DDSM dataset to prepare training data for RetinaNet.
It directly extracts bounding boxes from the mask values without normalizing to 0-255.
"""

import os
import pandas as pd
import numpy as np
import pydicom
import cv2
from pathlib import Path
import argparse
from tqdm import tqdm
import glob
import re

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Prepare DDSM data for RetinaNet training')
    parser.add_argument('--csv_file', default='d:/TCIA/manifest-ZkhPvrLo5216730872708713142/calc_case_description_test_set.csv',
                        help='Path to the case description CSV file')
    parser.add_argument('--ddsm_dir', default='d:/TCIA/manifest-ZkhPvrLo5216730872708713142/CBIS-DDSM',
                        help='Path to the CBIS-DDSM directory')
    parser.add_argument('--output_dir', default='ddsm_retinanet_data_calc_test2',
                        help='Output directory for prepared data')
    parser.add_argument('--type', default='calc', choices=['calc', 'mass'],
                        help='Type of abnormality to process (calc or mass)')
    parser.add_argument('--mode', default='train', choices=['train', 'test'],
                        help='Dataset mode (train or test)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit processing to N samples (for testing)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug outputs')
    return parser.parse_args()

def ensure_dir(directory):
    """Create directory if it doesn't exist."""
    os.makedirs(directory, exist_ok=True)
    return directory

def find_dicom_files(ddsm_dir, patient_prefix, view_type, is_roi_mask=False):
    """Find DICOM files for a specific patient and view, optionally for ROI masks."""
    search_pattern = os.path.join(ddsm_dir, f"{patient_prefix}*")
    matching_dirs = glob.glob(search_pattern)
    
    if not matching_dirs:
        return None
    
    # Extract files from matching directories
    for directory in matching_dirs:
        # Walk through subdirectories to find DICOM files
        for root, dirs, files in os.walk(directory):
            # Look for ROI mask images or full mammogram images
            target_term = "ROI mask" if is_roi_mask else "full mammogram"
            if target_term in root:
                dicom_files = [os.path.join(root, f) for f in files if f.endswith('.dcm')]
                if dicom_files:
                    # For ROI masks, we want the segmentation file (usually 1-2.dcm if present)
                    if is_roi_mask:
                        # Try to find the segmentation file first
                        seg_files = [f for f in dicom_files if f.endswith('1-2.dcm')]
                        if seg_files:
                            return seg_files[0]
                        # Fall back to any DCM in the ROI mask directory
                        return dicom_files[0]
                    # Otherwise, return the first full mammogram image
                    return dicom_files[0]
    
    return None

def load_dicom_image(dicom_path):
    """Load a DICOM image."""
    try:
        dicom = pydicom.dcmread(dicom_path)
        pixel_array = dicom.pixel_array
        return pixel_array
    except Exception as e:
        print(f"Error loading DICOM {dicom_path}: {e}")
        return None

def load_dicom_image_normalized(dicom_path):
    """Load and normalize a DICOM image for display."""
    try:
        dicom = pydicom.dcmread(dicom_path)
        pixel_array = dicom.pixel_array
        
        # Normalize to 8-bit (0-255)
        if pixel_array.max() > 255:
            pixel_array = ((pixel_array - pixel_array.min()) / 
                          (pixel_array.max() - pixel_array.min()) * 255).astype(np.uint8)
        else:
            pixel_array = pixel_array.astype(np.uint8)
            
        return pixel_array
    except Exception as e:
        print(f"Error loading and normalizing DICOM {dicom_path}: {e}")
        return None

def extract_bounding_box_from_mask(mask_array):
    """Extract bounding box from mask values (0=background, 255=foreground)."""
    # Find pixels with value 255 (or greater than a threshold)
    mask_binary = mask_array > 127  # Use threshold in case values aren't exactly 255
    
    # Find non-zero rows and columns
    rows = np.any(mask_binary, axis=1)
    cols = np.any(mask_binary, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        return None
    
    # Get min/max indices
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    # Return x1, y1, x2, y2 format
    return int(cmin), int(rmin), int(cmax), int(rmax)

def save_as_jpg(image, output_path):
    """Save image array as JPG."""
    try:
        cv2.imwrite(output_path, image)
        return True
    except Exception as e:
        print(f"Error saving JPG {output_path}: {e}")
        return False

def process_case(row, ddsm_dir, output_dir, abnorm_type, mode):
    """Process a single case from the CSV."""
    # Extract patient info
    patient_id = row['patient_id']
    breast_side = row['left or right breast']
    view = row['image view']
    abnorm_id = row['abnormality id']
    pathology = row['pathology']
    
    # Create search patterns based on abnormality type and mode
    type_prefix = 'Calc' if abnorm_type == 'calc' else 'Mass'
    mode_prefix = 'Training' if mode == 'train' else 'Test'
    
    # Create search pattern for finding corresponding files
    search_prefix = f"{type_prefix}-{mode_prefix}_{patient_id}_{breast_side}_{view}"
    roi_search_prefix = f"{type_prefix}-{mode_prefix}_{patient_id}_{breast_side}_{view}_{abnorm_id}"
    
    # Find image and mask DICOM files
    image_dicom = find_dicom_files(ddsm_dir, search_prefix, view)
    mask_dicom = find_dicom_files(ddsm_dir, roi_search_prefix, view, is_roi_mask=True)
    
    if not image_dicom or not mask_dicom:
        print(f"Could not find DICOM files for {search_prefix}")
        return None
    
    # Load DICOM files - normalized image for saving as JPG
    normalized_image = load_dicom_image_normalized(image_dicom)
    # Raw mask for bounding box extraction
    raw_mask = load_dicom_image(mask_dicom)
    
    if normalized_image is None or raw_mask is None:
        return None
    
    # Print mask values to debug
    if args.debug:
        unique_values = np.unique(raw_mask)
        print(f"Mask unique values: {unique_values}, shape: {raw_mask.shape}, min: {raw_mask.min()}, max: {raw_mask.max()}")
    
    # Create output paths
    image_filename = f"{patient_id}_{breast_side}_{view}_{abnorm_id}.jpg"
    jpg_path = os.path.join(output_dir, 'images', image_filename)
    
    # Save image as JPG
    ensure_dir(os.path.dirname(jpg_path))
    if not save_as_jpg(normalized_image, jpg_path):
        return None
    
    # Extract bounding box directly from mask values
    bbox = extract_bounding_box_from_mask(raw_mask)
    
    if bbox is None:
        print(f"No bounding box found for {image_filename}")
        return None
    
    # Validate bounding box size
    x1, y1, x2, y2 = bbox
    if (x2 - x1) < 5 or (y2 - y1) < 5:
        print(f"Bounding box too small for {image_filename}: {bbox}")
        return None
    
    # Convert absolute path to relative path for CSV
    rel_jpg_path = os.path.relpath(jpg_path, os.path.dirname(output_dir))
    
    # Set class based on pathology and abnormality type
    if abnorm_type == 'calc':
        class_name = f"malignant calcification" if pathology == 'MALIGNANT' else f"benign calcification"
    else:
        class_name = f"malignant mass" if pathology == 'MALIGNANT' else f"benign mass"
    
    # Debug: save the mask with bounding box for verification
    if args.debug:
        debug_dir = ensure_dir(os.path.join(output_dir, 'debug'))
        # Convert mask to visible image 
        visible_mask = np.zeros_like(raw_mask)
        visible_mask[raw_mask > 127] = 255  # Use threshold if needed
        # Draw bounding box
        debug_mask = cv2.cvtColor(visible_mask, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(debug_mask, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Save debug image
        debug_path = os.path.join(debug_dir, f"debug_{image_filename}")
        cv2.imwrite(debug_path, debug_mask)
        # Also save the original mask for comparison
        orig_mask_path = os.path.join(debug_dir, f"mask_{image_filename}")
        cv2.imwrite(orig_mask_path, raw_mask)
    
    # Return annotation in required format: path,x1,y1,x2,y2,class
    return rel_jpg_path, *bbox, class_name

def main():
    global args
    args = parse_args()
    
    # Create output directories
    output_dir = ensure_dir(args.output_dir)
    ensure_dir(os.path.join(output_dir, 'images'))
    
    # Create appropriate class mapping file with both mass and calcification classes
    class_map_path = os.path.join(output_dir, 'class_map.csv')
    with open(class_map_path, 'w') as f:
        f.write('benign mass,0\nmalignant mass,1\nbenign calcification,2\nmalignant calcification,3\n')
    
    # Determine CSV filename pattern based on args
    csv_pattern = f"{args.type}_case_description_{args.mode}_set.csv"
    csv_file = args.csv_file if os.path.isfile(args.csv_file) else None
    
    # If the provided CSV doesn't exist, try to find it based on pattern
    if not csv_file:
        csv_search = os.path.join(os.path.dirname(args.ddsm_dir), f"*{csv_pattern}")
        matching_csvs = glob.glob(csv_search)
        if matching_csvs:
            csv_file = matching_csvs[0]
        else:
            print(f"Could not find CSV file matching pattern: {csv_pattern}")
            return
    
    # Load CSV file
    print(f"Loading CSV file: {csv_file}")
    df = pd.read_csv(csv_file)
    
    if args.limit is not None:
        df = df.head(args.limit)
    
    print(f"Processing {len(df)} {args.type} cases from CSV...")
    annotations = []
    
    # Process each row
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # Use the correct type when processing
        result = process_case(row, args.ddsm_dir, output_dir, args.type, args.mode)
        if result:
            annotations.append(result)
    
    # Write annotations file
    annotations_path = os.path.join(output_dir, 'annotations.csv')
    with open(annotations_path, 'w') as f:
        for annotation in annotations:
            f.write(','.join(map(str, annotation)) + '\n')
    
    print(f"Successfully processed {len(annotations)} {args.type} images")
    print(f"Annotations saved to {annotations_path}")
    print(f"Class map saved to {class_map_path}")
    print("\nTo train the RetinaNet model, use:")
    print(f"python train.py --dataset csv --csv_train {annotations_path} --csv_classes {class_map_path} --depth 50")

if __name__ == '__main__':
    main()
