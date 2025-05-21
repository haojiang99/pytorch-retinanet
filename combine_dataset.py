#!/usr/bin/env python3
"""
combine_dataset.py - Combines DDSM mass and calcification datasets into a unified dataset.
"""

import os
import shutil
import argparse
from tqdm import tqdm

# Define source and destination directories
SOURCE_DIRS = [
    'ddsm_retinanet_data_mass_train2',
    'ddsm_retinanet_data_mass_test2',
    'ddsm_retinanet_data_calc_train2',
    'ddsm_retinanet_data_calc_test2'
]
DEST_DIR = 'ddsm_train3'

# New class mapping
NEW_CLASS_MAP = {
    'benign mass': 0,
    'malignant mass': 1,
    'benign calcification': 2,
    'malignant calcification': 3
}

def parse_args():
    parser = argparse.ArgumentParser(description='Combine DDSM datasets')
    parser.add_argument('--output', default=DEST_DIR, help='Output directory')
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
        raise ValueError(f"Cannot determine data type from: {dir_name}")

def find_image_file(img_path, source_dir):
    """Find the actual image file by checking multiple possible paths."""
    # Try direct path
    full_path = os.path.join(source_dir, img_path)
    if os.path.exists(full_path):
        return full_path
    
    # Try without 'ddsm_retinanet_data' prefix
    if 'ddsm_retinanet_data' in img_path:
        simplified_path = img_path.replace('ddsm_retinanet_data/', '')
        full_path = os.path.join(source_dir, simplified_path)
        if os.path.exists(full_path):
            return full_path
    
    # Try just using the image filename
    img_filename = os.path.basename(img_path)
    potential_paths = [
        os.path.join(source_dir, img_filename),
        os.path.join(source_dir, 'images', img_filename)
    ]
    
    for path in potential_paths:
        if os.path.exists(path):
            return path
    
    return None

def copy_and_convert_annotations(source_dirs, dest_dir):
    """Copy images and convert annotations."""
    ensure_dir(dest_dir)
    images_dir = os.path.join(dest_dir, 'images')
    ensure_dir(images_dir)
    
    # Prepare new annotations file
    new_annotations_file = os.path.join(dest_dir, 'annotations.csv')
    new_annotations = []
    processed_count = 0
    
    # Process each source directory
    for source_dir in source_dirs:
        if not os.path.exists(source_dir):
            print(f"Warning: Source directory {source_dir} not found, skipping.")
            continue
        
        # Determine data type (mass or calcification)
        data_type = get_type_from_dir(source_dir)
        print(f"Processing {source_dir} ({data_type} data)...")
                
        # Read annotations
        annotations_path = os.path.join(source_dir, 'annotations.csv')
        try:
            with open(annotations_path, 'r') as f:
                annotations = f.readlines()
                
            for line in tqdm(annotations, desc=f"Processing {source_dir}"):
                if not line.strip():
                    continue
                    
                parts = line.strip().split(',')
                if len(parts) == 6:  # path,x1,y1,x2,y2,class_name
                    img_path = parts[0]
                    x1, y1, x2, y2 = parts[1:5]
                    class_name = parts[5]  # String like 'benign' or 'malignant'
                    
                    # Create new class name with type and pathology
                    # Check if class_name already includes the data_type to avoid duplication
                    if data_type in class_name:
                        new_class_name = class_name
                    else:
                        new_class_name = f"{class_name} {data_type}"
                    
                    # Find the actual image file
                    src_img_path = find_image_file(img_path, source_dir)
                    
                    if src_img_path:
                        # Create unique name to avoid overwriting
                        base_name = os.path.basename(src_img_path)
                        unique_name = f"{data_type}_{base_name}"
                        dest_img_path = os.path.join(images_dir, unique_name)
                        
                        # Copy the image file
                        shutil.copy2(src_img_path, dest_img_path)
                        processed_count += 1
                        
                        # Add to new annotations with updated path and class
                        # Use absolute path instead of relative path
                        abs_img_path = os.path.abspath(dest_img_path).replace('\\', '/')
                        
                        # Use the class name, not the ID for the annotation
                        new_annotation = f"{abs_img_path},{x1},{y1},{x2},{y2},{new_class_name}"
                        new_annotations.append(new_annotation)
                    else:
                        print(f"Warning: Image not found: {img_path} in {source_dir}")
                        
        except Exception as e:
            print(f"Error reading annotations from {annotations_path}: {e}")
            continue
    
    # Write new annotations file
    with open(new_annotations_file, 'w') as f:
        for annotation in new_annotations:
            f.write(annotation + '\n')
    
    print(f"Processed {processed_count} images")
    print(f"Created {len(new_annotations)} combined annotations in {new_annotations_file}")
    
    # Write new class map
    new_class_map_file = os.path.join(dest_dir, 'class_map.csv')
    with open(new_class_map_file, 'w') as f:
        for class_name, class_id in NEW_CLASS_MAP.items():
            f.write(f"{class_name},{class_id}\n")
    
    print(f"Created new class map in {new_class_map_file}")

def main():
    args = parse_args()
    dest_dir = args.output
    
    print(f"Combining datasets from {SOURCE_DIRS} into {dest_dir}")
    
    # Process all source directories
    copy_and_convert_annotations(SOURCE_DIRS, dest_dir)
    
    print(f"Dataset combination complete. Combined dataset is in {dest_dir}")
    print(f"New class mapping:")
    for class_name, class_id in sorted(NEW_CLASS_MAP.items(), key=lambda x: x[1]):
        print(f"  {class_id}: {class_name}")

if __name__ == '__main__':
    main()
