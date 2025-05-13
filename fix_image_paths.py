#!/usr/bin/env python3
"""
fix_image_paths.py - Fixes image paths in annotation files
"""

import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Fix image paths in annotations')
    parser.add_argument('--dataset', default='ddsm_train_mini', help='Dataset directory')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # File paths
    dataset_dir = args.dataset
    annotations_path = os.path.join(dataset_dir, 'annotations.csv')
    
    # Check files exist
    if not os.path.exists(annotations_path):
        print(f"Annotations file not found: {annotations_path}")
        return
        
    # Read annotations
    with open(annotations_path, 'r') as f:
        annotations = f.readlines()
    
    # Fix image paths
    fixed_annotations = []
    for line in annotations:
        if not line.strip():
            continue
        
        parts = line.strip().split(',')
        if len(parts) >= 6:  # path,x1,y1,x2,y2,class_name
            img_path = parts[0]
            
            # Remove dataset directory prefix from image path if present
            if dataset_dir in img_path:
                img_path = img_path.replace(f"{dataset_dir}/", "")
                img_path = img_path.replace(f"{dataset_dir}\\", "")
                
            # Make sure path uses correct path separators
            img_path = img_path.replace('/', os.sep)
            
            # Check if image exists
            full_path = os.path.join(dataset_dir, img_path)
            if not os.path.exists(full_path):
                print(f"Warning: Image not found: {full_path}")
                # No need to fix if the image doesn't exist
                fixed_annotations.append(line.strip())
                continue
                
            # Create updated line with fixed path
            fixed_line = f"{img_path},{','.join(parts[1:])}"
            fixed_annotations.append(fixed_line)
        else:
            # Keep original line if it doesn't have the expected format
            fixed_annotations.append(line.strip())
    
    # Backup original file
    backup_path = annotations_path + '.path_bak'
    os.rename(annotations_path, backup_path)
    print(f"Original annotations backed up to {backup_path}")
    
    # Write fixed annotations
    with open(annotations_path, 'w') as f:
        for annotation in fixed_annotations:
            f.write(annotation + '\n')
    
    print(f"Fixed {len(fixed_annotations)} annotations")
    print("Done!")

if __name__ == '__main__':
    main()
