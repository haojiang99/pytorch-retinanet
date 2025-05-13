#!/usr/bin/env python3
"""
verify_dataset.py - Verifies and fixes dataset annotations
"""

import os
import argparse
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description='Verify and fix dataset')
    parser.add_argument('--dataset', default='ddsm_train_mini', help='Dataset directory')
    return parser.parse_args()

def main():
    args = parse_args()
    
    dataset_dir = args.dataset
    annotations_path = os.path.join(dataset_dir, 'annotations.csv')
    class_map_path = os.path.join(dataset_dir, 'class_map.csv')
    
    if not os.path.exists(annotations_path):
        print(f"Error: Annotations file not found: {annotations_path}")
        return
    if not os.path.exists(class_map_path):
        print(f"Error: Class map file not found: {class_map_path}")
        return
    
    # Load class map
    class_map = {}
    name_to_id = {}
    with open(class_map_path, 'r') as f:
        for line in f:
            if line.strip():
                class_name, class_id = line.strip().split(',')
                class_map[class_id] = class_name
                name_to_id[class_name] = class_id
    
    print(f"Class map: {class_map}")
    
    # Read annotations
    with open(annotations_path, 'r') as f:
        annotations = f.readlines()
    
    valid_annotations = []
    for line in annotations:
        if not line.strip():
            continue
        
        parts = line.strip().split(',')
        if len(parts) >= 6:
            img_path, x1, y1, x2, y2, class_name = parts[:6]
            
            # Check if the class name is a number (class ID)
            if class_name.isdigit() and class_name in class_map:
                class_name = class_map[class_name]
                print(f"Fixed class ID to name: {parts[5]} -> {class_name}")
            
            # Ensure class name is valid
            if class_name not in name_to_id:
                print(f"Warning: Invalid class name: {class_name}")
                continue
            
            # Fix the image path
            full_path = os.path.join(dataset_dir, img_path)
            
            # Try to find the image 
            if not os.path.exists(full_path):
                # Try some variations
                if img_path.startswith("images\\") or img_path.startswith("images/"):
                    # Path is already relative, try without "images\" prefix
                    alt_path = os.path.join(dataset_dir, os.path.basename(img_path))
                    if os.path.exists(alt_path):
                        full_path = alt_path
                        img_path = os.path.basename(img_path)
                        print(f"Fixed path: {parts[0]} -> {img_path}")
                else:
                    # Try with "images\" prefix
                    alt_path = os.path.join(dataset_dir, "images", os.path.basename(img_path))
                    if os.path.exists(alt_path):
                        full_path = alt_path
                        img_path = os.path.join("images", os.path.basename(img_path))
                        print(f"Fixed path: {parts[0]} -> {img_path}")
            
            # Verify the image exists
            if not os.path.exists(full_path):
                print(f"Warning: Image not found: {full_path}")
                continue
            
            # Verify the image can be opened
            try:
                with Image.open(full_path) as img:
                    width, height = img.size
                    
                    # Verify bounding box coordinates are valid
                    x1_val, y1_val, x2_val, y2_val = int(float(x1)), int(float(y1)), int(float(x2)), int(float(y2))
                    if x1_val >= x2_val or y1_val >= y2_val:
                        print(f"Warning: Invalid box coordinates: {x1_val},{y1_val},{x2_val},{y2_val}")
                        continue
                    if x1_val < 0 or y1_val < 0 or x2_val > width or y2_val > height:
                        print(f"Warning: Box outside image bounds: {x1_val},{y1_val},{x2_val},{y2_val} for image {width}x{height}")
                        # Fix box to be within image bounds
                        x1_val = max(0, x1_val)
                        y1_val = max(0, y1_val)
                        x2_val = min(width, x2_val)
                        y2_val = min(height, y2_val)
                        print(f"Fixed to: {x1_val},{y1_val},{x2_val},{y2_val}")
                        x1, y1, x2, y2 = str(x1_val), str(y1_val), str(x2_val), str(y2_val)
            except Exception as e:
                print(f"Warning: Failed to open image {full_path}: {e}")
                continue
            
            # Add valid annotation
            valid_line = f"{img_path},{x1},{y1},{x2},{y2},{class_name}"
            valid_annotations.append(valid_line)
    
    # Backup original file
    backup_path = annotations_path + '.verified_bak'
    os.rename(annotations_path, backup_path)
    print(f"Original annotations backed up to {backup_path}")
    
    # Write verified annotations
    with open(annotations_path, 'w') as f:
        for annotation in valid_annotations:
            f.write(annotation + '\n')
    
    print(f"Saved {len(valid_annotations)} valid annotations out of {len(annotations)} total")
    print("Dataset verification complete!")

if __name__ == '__main__':
    main()
