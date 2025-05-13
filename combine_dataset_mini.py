#!/usr/bin/env python3
"""
combine_dataset_mini.py - Creates a mini combined dataset with 50 images from each source.
"""

import os
import shutil
import argparse
import random
from tqdm import tqdm

# Define source and destination directories
SOURCE_DIRS = [
    'ddsm_retinanet_data_mass_train',
    'ddsm_retinanet_data_mass_test',
    'ddsm_retinanet_data_calc_train',
    'ddsm_retinanet_data_calc_test'
]
DEST_DIR = 'ddsm_train_mini'
SAMPLE_SIZE = 50  # Number of samples to take from each source

# New class mapping
NEW_CLASS_MAP = {
    'benign mass': 0,
    'malignant mass': 1,
    'benign calcification': 2,
    'malignant calcification': 3
}

def parse_args():
    parser = argparse.ArgumentParser(description='Create mini combined DDSM dataset')
    parser.add_argument('--output', default=DEST_DIR, help='Output directory')
    parser.add_argument('--samples', type=int, default=SAMPLE_SIZE, help='Number of samples per source')
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

def create_mini_dataset(source_dirs, dest_dir, samples_per_source):
    """Create a mini dataset with a specified number of samples from each source."""
    ensure_dir(dest_dir)
    images_dir = os.path.join(dest_dir, 'images')
    ensure_dir(images_dir)
    
    # Prepare new annotations file
    new_annotations_file = os.path.join(dest_dir, 'annotations.csv')
    new_annotations = []
    
    # Process each source directory
    total_processed = 0
    
    for source_dir in source_dirs:
        if not os.path.exists(source_dir):
            print(f"Warning: Source directory {source_dir} not found, skipping.")
            continue
        
        # Determine data type (mass or calcification)
        data_type = get_type_from_dir(source_dir)
        print(f"Processing {source_dir} ({data_type} data)...")
                
        # Read annotations
        annotations_path = os.path.join(source_dir, 'annotations.csv')
        valid_annotations = []
        
        try:
            with open(annotations_path, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue
                        
                    parts = line.strip().split(',')
                    if len(parts) == 6:  # path,x1,y1,x2,y2,class_name
                        img_path = parts[0]
                        src_img_path = find_image_file(img_path, source_dir)
                        
                        if src_img_path:
                            valid_annotations.append((parts, src_img_path))
            
            # Randomly sample from valid annotations
            sample_count = min(samples_per_source, len(valid_annotations))
            selected_annotations = random.sample(valid_annotations, sample_count)
            
            print(f"Selected {sample_count} samples from {len(valid_annotations)} valid annotations")
            
            # Process the selected annotations
            for (parts, src_img_path) in tqdm(selected_annotations, desc=f"Processing samples from {source_dir}"):
                img_path, x1, y1, x2, y2, class_name = parts
                
                # Create new class name with type and pathology
                new_class_name = f"{class_name} {data_type}"
                
                # Create unique name to avoid overwriting
                base_name = os.path.basename(src_img_path)
                unique_name = f"{data_type}_{base_name}"
                dest_img_path = os.path.join(images_dir, unique_name)
                
                # Copy the image file
                shutil.copy2(src_img_path, dest_img_path)
                total_processed += 1
                
                # Add to new annotations with updated path and class
                new_img_path = os.path.join('images', unique_name)
                new_class_id = NEW_CLASS_MAP.get(new_class_name, -1)
                
                if new_class_id != -1:
                    new_annotation = f"{new_img_path},{x1},{y1},{x2},{y2},{new_class_id}"
                    new_annotations.append(new_annotation)
                else:
                    print(f"Warning: Unknown class: {new_class_name}")
                        
        except Exception as e:
            print(f"Error processing {annotations_path}: {e}")
            continue
    
    # Write new annotations file
    with open(new_annotations_file, 'w') as f:
        for annotation in new_annotations:
            f.write(annotation + '\n')
    
    print(f"Total images processed: {total_processed}")
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
    samples = args.samples
    
    print(f"Creating mini dataset with {samples} samples per source")
    print(f"Source directories: {SOURCE_DIRS}")
    print(f"Destination: {dest_dir}")
    
    # Create mini dataset
    create_mini_dataset(SOURCE_DIRS, dest_dir, samples)
    
    print(f"Mini dataset creation complete. Dataset is in {dest_dir}")
    print(f"Class mapping:")
    for class_name, class_id in sorted(NEW_CLASS_MAP.items(), key=lambda x: x[1]):
        print(f"  {class_id}: {class_name}")

if __name__ == '__main__':
    main()
