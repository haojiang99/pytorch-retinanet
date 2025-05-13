#!/usr/bin/env python3
"""
absolute_path_fix.py - Fixes paths in annotations to use absolute paths
"""

import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Fix paths in annotations')
    parser.add_argument('--dataset', default='ddsm_train_mini', help='Dataset directory')
    return parser.parse_args()

def main():
    args = parse_args()
    
    dataset_dir = os.path.abspath(args.dataset)
    annotations_path = os.path.join(dataset_dir, 'annotations.csv')
    
    print(f"Using absolute path: {dataset_dir}")
    
    # Read annotations
    with open(annotations_path, 'r') as f:
        annotations = f.readlines()
    
    fixed_annotations = []
    for line in annotations:
        if not line.strip():
            continue
        
        parts = line.strip().split(',')
        if len(parts) >= 6:  # path,x1,y1,x2,y2,class_name
            img_path = parts[0]
            remainder = ','.join(parts[1:])
            
            # Extract filename from path
            filename = os.path.basename(img_path)
            
            # Try different paths
            potential_paths = [
                os.path.join(dataset_dir, img_path),
                os.path.join(dataset_dir, "images", filename)
            ]
            
            found_path = None
            for path in potential_paths:
                if os.path.exists(path):
                    found_path = path
                    break
            
            if found_path:
                fixed_line = f"{found_path},{remainder}"
                fixed_annotations.append(fixed_line)
                print(f"Fixed: {img_path} -> {found_path}")
            else:
                print(f"Warning: Image not found: {img_path}")
                # Keep the line but the training will likely fail
                fixed_annotations.append(line.strip())
        else:
            fixed_annotations.append(line.strip())
    
    # Backup original file
    backup_path = annotations_path + '.abs_bak'
    os.rename(annotations_path, backup_path)
    print(f"Original annotations backed up to {backup_path}")
    
    # Write fixed annotations
    with open(annotations_path, 'w') as f:
        for annotation in fixed_annotations:
            f.write(annotation + '\n')
    
    print(f"Fixed {len(fixed_annotations)} annotations with absolute paths")

if __name__ == '__main__':
    main()
