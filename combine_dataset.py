#!/usr/bin/env python3
"""
combine_dataset.py

Script to combine DDSM mass and calcification datasets into a single dataset
with four classes: malignant mass, malignant calcification, benign mass, and benign calcification.
"""

import os
import shutil
import argparse
import csv
from tqdm import tqdm

# Define source and destination directories
SOURCE_DIRS = [
    'ddsm_retinanet_data_mass_train',
    'ddsm_retinanet_data_mass_test',
    'ddsm_retinanet_data_calc_train',
    'ddsm_retinanet_data_calc_test'
]
DEST_DIR = 'ddsm_train'
DEST_IMAGES_DIR = os.path.join(DEST_DIR, 'images')

# New class mapping
NEW_CLASS_MAP = {
    'benign mass': 0,
    'malignant mass': 1,
    'benign calcification': 2,
    'malignant calcification': 3
}

def parse_args():
    parser = argparse.ArgumentParser(description='Combine DDSM mass and calcification datasets')
    parser.add_argument('--output', default=DEST_DIR, help='Output directory (default: ddsm_train)')
    return parser.parse_args()

def ensure_dir(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

def get_type_from_dir(dir_name):
    """Determine if directory contains mass or calcification data."""
    if 'mass' in dir_name.lower():
        return 'mass'
    elif 'calc' in dir_name.lower():
        return 'calcification'
    else:
        raise ValueError(f"Cannot determine data type from directory name: {dir_name}")

def copy_and_convert_annotations(source_dirs, dest_dir):
    """
    Copy images and convert annotations from source directories to destination directory.
    Update class names to include both type (mass/calcification) and pathology (benign/malignant).
    """
    ensure_dir(dest_dir)
    images_dir = os.path.join(dest_dir, 'images')
    ensure_dir(images_dir)
    
    # Prepare new annotations file
    new_annotations_file = os.path.join(dest_dir, 'annotations.csv')
    new_annotations = []
    
    # Process each source directory
    for source_dir in source_dirs:
        if not os.path.exists(source_dir):
            print(f"Warning: Source directory {source_dir} not found, skipping.")
            continue
        
        # Determine if the source directory contains mass or calcification data
        data_type = get_type_from_dir(source_dir)
        print(f"Processing {source_dir} ({data_type} data)...")
        
        # Read class map to determine class names
        class_map_path = os.path.join(source_dir, 'class_map.csv')
        class_map = {}
        try:
            with open(class_map_path, 'r') as f:
                for line in f:
                    if line.strip():
                        class_name, class_id = line.strip().split(',')
                        class_map[int(class_id)] = class_name
        except Exception as e:
            print(f"Error reading class map from {class_map_path}: {e}")
            continue
        
        # Read annotations
        annotations_path = os.path.join(source_dir, 'annotations.csv')
        try:
            with open(annotations_path, 'r') as f:
                for line in f:
                    if line.strip():
                        parts = line.strip().split(',')
                        if len(parts) == 6:  # path,x1,y1,x2,y2,class_id
                            img_path = parts[0]
                            x1, y1, x2, y2 = parts[1:5]
                            class_id = int(parts[5])
                            
                            # Get original class name (benign/malignant)
                            old_class_name = class_map.get(class_id, f"unknown-{class_id}")
                            
                            # Create new class name with type and pathology
                            new_class_name = f"{old_class_name} {data_type}"
                            
                            # Copy image to destination
                            src_img_path = os.path.join(source_dir, img_path)
                            base_name = os.path.basename(img_path)
                            
                            # Create unique name to avoid overwriting
                            unique_name = f"{data_type}_{base_name}"
                            dest_img_path = os.path.join(images_dir, unique_name)
                            
                            # Copy the image file if it exists
                            if os.path.exists(src_img_path):
                                shutil.copy2(src_img_path, dest_img_path)
                                
                                # Add to new annotations with updated path and class
                                new_img_path = os.path.join('images', unique_name)
                                new_class_id = NEW_CLASS_MAP.get(new_class_name, -1)
                                
                                if new_class_id != -1:
                                    new_annotation = f"{new_img_path},{x1},{y1},{x2},{y2},{new_class_id}"
                                    new_annotations.append(new_annotation)
                                else:
                                    print(f"Warning: Unknown new class name: {new_class_name}")
                            else:
                                print(f"Warning: Image file not found: {src_img_path}")
        except Exception as e:
            print(f"Error reading annotations from {annotations_path}: {e}")
            continue
    
    # Write new annotations file
    with open(new_annotations_file, 'w') as f:
        for annotation in new_annotations:
            f.write(annotation + '\n')
    
    print(f"Created {len(new_annotations)} combined annotations in {new_annotations_file}")
    
    # Write new class map
    new_class_map_file = os.path.join(dest_dir, 'class_map.csv')
    with open(new_class_map_file, 'w') as f:
        for class_name, class_id in NEW_CLASS_MAP.items():
            f.write(f"{class_name},{class_id}\n")
    
    print(f"Created new class map in {new_class_map_file}")

def main():
    args = parse_args()
    
    # Update destination directory if specified via command line
    dest_dir = args.output
    
    print(f"Combining datasets from {SOURCE_DIRS} into {dest_dir}")
    
    # Process all source directories
    copy_and_convert_annotations(SOURCE_DIRS, dest_dir)
    
    print(f"Dataset combination complete. Combined dataset is in {dest_dir}")
    print(f"New class mapping:")
    for class_name, class_id in NEW_CLASS_MAP.items():
        print(f"  {class_id}: {class_name}")

if __name__ == '__main__':
    main()
