#!/usr/bin/env python3
"""
DDSM Data Visualization Script

This script visualizes the prepared mammogram mass data by drawing bounding boxes
on the original images and saving them to a separate folder.
"""

import os
import pandas as pd
import cv2
import argparse
from tqdm import tqdm

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Visualize prepared DDSM data')
    parser.add_argument('--annotations', default='ddsm_retinanet_data_calc_test2/annotations.csv',
                        help='Path to the annotations CSV file')
    parser.add_argument('--class_map', default='ddsm_retinanet_data_calc_test2/class_map.csv',
                        help='Path to the class_map CSV file')
    parser.add_argument('--output_dir', default='ddsm_retinanet_data_calc_test2_visual',
                        help='Output directory for visualization images')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit visualization to N samples')
    return parser.parse_args()

def ensure_dir(directory):
    """Create directory if it doesn't exist."""
    os.makedirs(directory, exist_ok=True)
    return directory

def load_class_map(class_map_path):
    """Load class mapping from CSV file."""
    class_map = {}
    with open(class_map_path, 'r') as f:
        for line in f:
            if line.strip():
                class_name, class_id = line.strip().split(',')
                class_map[class_name] = int(class_id)
    # Create reverse mapping (id to name)
    id_to_name = {v: k for k, v in class_map.items()}
    return class_map, id_to_name

def visualize_annotation(image_path, x1, y1, x2, y2, class_name, output_path):
    """Draw bounding box on image and save it."""
    try:
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image at {image_path}")
            return False
        
        # Get image dimensions
        h, w = image.shape[:2]
        
        # Ensure coordinates are within image boundaries
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        
        # Draw bounding box with thicker lines (4 pixels)
        color = (0, 255, 0) if class_name == 'benign' else (0, 0, 255)  # Green for benign, Red for malignant
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 10)
        
        # Draw a contrasting outline to make the box more visible
        cv2.rectangle(image, (x1-1, y1-1), (x2+1, y2+1), (255, 255, 255), 2)
        
        # Add label with improved visibility
        label = f"{class_name.upper()}"
        
        # Measure text size to create better background box
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
        
        # Draw filled rectangle for text background
        text_box_y1 = max(0, y1 - text_size[1] - 10)
        text_box_x1 = max(0, x1 - 2)
        text_box_x2 = min(w, x1 + text_size[0] + 5)
        
        cv2.rectangle(image, 
                     (text_box_x1, text_box_y1), 
                     (text_box_x2, y1), 
                     (0, 0, 0), 
                     -1)  # Filled rectangle
        
        # Add white border to the background box
        cv2.rectangle(image, 
                     (text_box_x1, text_box_y1), 
                     (text_box_x2, y1), 
                     (255, 255, 255), 
                     1)  # White border
        
        # Add text with larger font and thicker lines
        cv2.putText(image, 
                   label, 
                   (x1, y1 - 5),  # Position adjusted for better placement
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   3,  # Larger font size (was 0.9)
                   (255, 255, 255),  # White text
                   3)  # Thicker font (was 2)
        
        # Save the image
        ensure_dir(os.path.dirname(output_path))
        cv2.imwrite(output_path, image)
        return True
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return False

def main():
    args = parse_args()
    
    # Create output directory
    output_dir = ensure_dir(args.output_dir)
    
    # Load class mapping
    _, id_to_name = load_class_map(args.class_map)
    
    # Read annotations
    try:
        annotations = []
        with open(args.annotations, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split(',')
                    if len(parts) == 6:  # path,x1,y1,x2,y2,class
                        annotations.append({
                            'path': parts[0],
                            'x1': int(parts[1]),
                            'y1': int(parts[2]),
                            'x2': int(parts[3]),
                            'y2': int(parts[4]),
                            'class': parts[5]
                        })
    except Exception as e:
        print(f"Error reading annotations: {e}")
        return
    
    # Limit if specified
    if args.limit is not None:
        annotations = annotations[:args.limit]
    
    print(f"Visualizing {len(annotations)} images...")
    success_count = 0
    
    # Process each annotation
    for ann in tqdm(annotations):
        # Extract annotation data
        img_path = ann['path']
        x1, y1, x2, y2 = ann['x1'], ann['y1'], ann['x2'], ann['y2']
        class_name = ann['class']
        
        # Determine output path
        base_name = os.path.basename(img_path)
        output_path = os.path.join(output_dir, base_name)
        
        # Check if the base directory exists based on where the script is run
        if not os.path.exists(img_path) and os.path.exists(os.path.join('ddsm_retinanet_data_mass_test2', img_path)):
            img_path = os.path.join('ddsm_retinanet_data_mass_test2', img_path)
        
        # Visualize
        if visualize_annotation(img_path, x1, y1, x2, y2, class_name, output_path):
            success_count += 1
    
    print(f"Successfully visualized {success_count} out of {len(annotations)} images")
    print(f"Images saved to: {output_dir}")

if __name__ == '__main__':
    main()
