#!/usr/bin/env python3
"""
fix_annotations.py - Converts class IDs to class names in annotation files
"""

import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Fix annotation files')
    parser.add_argument('--dataset', default='ddsm_train_mini', help='Dataset directory')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # File paths
    dataset_dir = args.dataset
    annotations_path = os.path.join(dataset_dir, 'annotations.csv')
    class_map_path = os.path.join(dataset_dir, 'class_map.csv')
    
    # Check files exist
    if not os.path.exists(annotations_path):
        print(f"Annotations file not found: {annotations_path}")
        return
    if not os.path.exists(class_map_path):
        print(f"Class map file not found: {class_map_path}")
        return
    
    # Load class map (id to name)
    id_to_name = {}
    with open(class_map_path, 'r') as f:
        for line in f:
            if line.strip():
                class_name, class_id = line.strip().split(',')
                id_to_name[class_id] = class_name
    
    print(f"Loaded class map: {id_to_name}")
    
    # Read annotations
    with open(annotations_path, 'r') as f:
        annotations = f.readlines()
    
    # Convert class IDs to names
    fixed_annotations = []
    for line in annotations:
        if not line.strip():
            continue
        
        parts = line.strip().split(',')
        if len(parts) == 6:  # path,x1,y1,x2,y2,class_id
            img_path, x1, y1, x2, y2, class_id = parts
            
            # Replace class ID with class name
            if class_id in id_to_name:
                class_name = id_to_name[class_id]
                fixed_line = f"{img_path},{x1},{y1},{x2},{y2},{class_name}"
                fixed_annotations.append(fixed_line)
            else:
                print(f"Warning: Unknown class ID: {class_id}")
                fixed_annotations.append(line.strip())
    
    # Backup original file
    backup_path = annotations_path + '.bak'
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
