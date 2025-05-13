#!/usr/bin/env python3
"""
DDSM Data Preparation Script for RetinaNet Training

This script processes the calc_case_description_test_set.csv from DDSM dataset
to prepare training data for RetinaNet. It handles the directory structure
of the CBIS-DDSM dataset, extracts DICOM images, converts them to JPG, and 
generates bounding boxes from ROI masks.
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
    parser.add_argument('--csv_file', default='d:/TCIA/manifest-ZkhPvrLo5216730872708713142/calc_case_description_train_set.csv',
                        help='Path to the calc_case_description_test_set.csv file')
    parser.add_argument('--ddsm_dir', default='d:/TCIA/manifest-ZkhPvrLo5216730872708713142/CBIS-DDSM',
                        help='Path to the CBIS-DDSM directory')
    parser.add_argument('--output_dir', default='ddsm_retinanet_data_calc_train',
                        help='Output directory for prepared data')
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
    """Load and process a DICOM image."""
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
        print(f"Error loading DICOM {dicom_path}: {e}")
        return None

def extract_bounding_box_from_mask(mask_array):
    """Extract bounding box from a binary mask."""
    # Find non-zero pixels
    rows = np.any(mask_array, axis=1)
    cols = np.any(mask_array, axis=0)
    
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

def process_case(row, ddsm_dir, output_dir):
    """Process a single case from the CSV."""
    # Extract patient info
    patient_id = row['patient_id']
    breast_side = row['left or right breast']
    view = row['image view']
    abnorm_id = row['abnormality id']
    pathology = row['pathology']
    
    # Create search pattern for finding corresponding files
    search_prefix = f"Calc-Training_{patient_id}_{breast_side}_{view}"
    # search_prefix = f"Calc-Test_{patient_id}_{breast_side}_{view}"
    roi_search_prefix = f"Calc-Training_{patient_id}_{breast_side}_{view}_{abnorm_id}"
    # roi_search_prefix = f"Calc-Test_{patient_id}_{breast_side}_{view}_{abnorm_id}"
    
    # Find image and mask DICOM files
    image_dicom = find_dicom_files(ddsm_dir, search_prefix, view)
    mask_dicom = find_dicom_files(ddsm_dir, roi_search_prefix, view, is_roi_mask=True)
    
    if not image_dicom or not mask_dicom:
        print(f"Could not find DICOM files for {search_prefix}")
        return None
    
    # Load DICOM files
    image = load_dicom_image(image_dicom)
    mask = load_dicom_image(mask_dicom)
    
    if image is None or mask is None:
        return None
    
    # Create output paths
    image_filename = f"{patient_id}_{breast_side}_{view}_{abnorm_id}.jpg"
    jpg_path = os.path.join(output_dir, 'images', image_filename)
    
    # Save image as JPG
    ensure_dir(os.path.dirname(jpg_path))
    if not save_as_jpg(image, jpg_path):
        return None
    
    # Extract bounding box
    # Threshold the mask if needed
    if mask.max() > 1:
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    else:
        binary_mask = mask * 255
    
    bbox = extract_bounding_box_from_mask(binary_mask)
    
    if bbox is None:
        print(f"No bounding box found for {image_filename}")
        return None
    
    # Convert absolute path to relative path for CSV
    rel_jpg_path = os.path.relpath(jpg_path, os.path.dirname(output_dir))
    
    # Set class based on pathology
    class_name = 'malignant' if pathology == 'MALIGNANT' else 'benign'
    
    # Return annotation in required format: path,x1,y1,x2,y2,class
    return rel_jpg_path, *bbox, class_name

def main():
    args = parse_args()
    
    # Create output directories
    output_dir = ensure_dir(args.output_dir)
    ensure_dir(os.path.join(output_dir, 'images'))
    
    # Create class mapping file
    class_map_path = os.path.join(output_dir, 'class_map.csv')
    with open(class_map_path, 'w') as f:
        f.write('benign,0\nmalignant,1\n')
    
    # Load CSV file
    df = pd.read_csv(args.csv_file)
    
    if args.limit is not None:
        df = df.head(args.limit)
    
    print(f"Processing {len(df)} calcification cases from CSV...")
    annotations = []
    
    # Process each row
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        result = process_case(row, args.ddsm_dir, output_dir)
        if result:
            annotations.append(result)
    
    # Write annotations file
    annotations_path = os.path.join(output_dir, 'annotations.csv')
    with open(annotations_path, 'w') as f:
        for annotation in annotations:
            f.write(','.join(map(str, annotation)) + '\n')
    
    print(f"Successfully processed {len(annotations)} calcification images")
    print(f"Annotations saved to {annotations_path}")
    print(f"Class map saved to {class_map_path}")
    print("\nTo train the RetinaNet model, use:")
    print(f"python train.py --dataset csv --csv_train {annotations_path} --csv_classes {class_map_path} --depth 50")

if __name__ == '__main__':
    main()
