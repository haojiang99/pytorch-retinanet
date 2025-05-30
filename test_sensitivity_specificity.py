#!/usr/bin/env python3
"""
Sensitivity and Specificity Testing Script for Mammogram Detection

This script evaluates the performance of a trained RetinaNet model on a test dataset
by calculating sensitivity (true positive rate) and specificity (true negative rate)
for mammogram lesion detection.

Usage:
    python test_sensitivity_specificity.py --dataset_path ddsm_train3 --model_path model.pt --threshold 0.1
"""

import argparse
import os
import csv
import cv2
import numpy as np
import torch
from collections import defaultdict
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from retinanet import model
from retinanet.model import ResNet, Bottleneck, BasicBlock
from torch import serialization
from torch.nn.parallel import DataParallel

# Class mapping for DDSM dataset
DDSM_CLASSES = {
    0: 'benign mass', 
    1: 'malignant mass',
    2: 'benign calcification',
    3: 'malignant calcification'
}


def load_model(model_path):
    """Load RetinaNet model from checkpoint"""
    print(f"Loading model from: {model_path}")
    
    try:
        # Add safe globals for PyTorch 2.6+
        serialization.add_safe_globals([ResNet, Bottleneck, BasicBlock, DataParallel])
        
        # Try loading with weights_only=False first
        retinanet = torch.load(model_path, map_location='cpu', weights_only=False)
        print("Loaded full model from checkpoint")
        
        # If it's wrapped in DataParallel, get the module
        if isinstance(retinanet, torch.nn.DataParallel):
            print("Model was saved with DataParallel, extracting module...")
            retinanet = retinanet.module
            
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    # Set model to evaluation mode
    retinanet.eval()
    retinanet.training = False
    
    # Move to GPU if available
    if torch.cuda.is_available():
        retinanet = retinanet.cuda()
        print("Model moved to GPU")
    
    return retinanet


def preprocess_image(image_path):
    """Preprocess image for model inference"""
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at path: {image_path}")
    
    # Convert BGR to RGB
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Get original dimensions
    original_height, original_width = image.shape[:2]
    
    # Calculate scale for resizing
    min_side = 800
    max_side = 1333
    smallest_side = min(original_height, original_width)
    scale = min_side / smallest_side
    
    # Check if the largest side exceeds max_side
    largest_side = max(original_height, original_width)
    if largest_side * scale > max_side:
        scale = max_side / largest_side
    
    # Resize the image
    new_width = int(round(original_width * scale))
    new_height = int(round(original_height * scale))
    image = cv2.resize(image, (new_width, new_height))
    
    # Pad to be divisible by 32
    pad_h = 32 - new_height % 32
    pad_w = 32 - new_width % 32
    
    padded_image = np.zeros((new_height + pad_h, new_width + pad_w, 3)).astype(np.float32)
    padded_image[:new_height, :new_width, :] = image.astype(np.float32)
    
    # Normalize
    padded_image /= 255.0
    padded_image -= [0.485, 0.456, 0.406]
    padded_image /= [0.229, 0.224, 0.225]
    
    # Convert to tensor
    padded_image = np.expand_dims(padded_image, 0)
    padded_image = np.transpose(padded_image, (0, 3, 1, 2))
    tensor = torch.from_numpy(padded_image).float()
    
    return tensor, scale


def load_annotations(annotations_file):
    """Load ground truth annotations from CSV file"""
    annotations = defaultdict(list)
    
    with open(annotations_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 6:  # path,x1,y1,x2,y2,class_name
                image_path = row[0]
                
                # Skip empty annotations (negative examples)
                if row[1] == '' or row[2] == '' or row[3] == '' or row[4] == '' or row[5] == '':
                    continue
                
                try:
                    x1, y1, x2, y2 = map(float, row[1:5])
                    class_name = row[5].strip()
                    
                    annotations[image_path].append({
                        'bbox': [x1, y1, x2, y2],
                        'class': class_name
                    })
                except ValueError:
                    continue
    
    return annotations


def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two bounding boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0.0


def run_inference(model, image_tensor, score_threshold=0.1):
    """Run inference on a single image"""
    with torch.no_grad():
        if torch.cuda.is_available():
            image_tensor = image_tensor.cuda()
        
        try:
            scores, labels, boxes = model(image_tensor)
            
            # Filter by score threshold
            valid_indices = scores > score_threshold
            
            detections = []
            for i in range(len(scores)):
                if valid_indices[i]:
                    detections.append({
                        'bbox': boxes[i].cpu().numpy(),
                        'score': scores[i].cpu().numpy(),
                        'class': DDSM_CLASSES.get(int(labels[i].cpu().numpy()), 'unknown')
                    })
            
            return detections
            
        except Exception as e:
            print(f"Error during inference: {e}")
            return []


def evaluate_performance(dataset_path, model_path, score_threshold=0.1, iou_threshold=0.5):
    """Evaluate model performance and calculate sensitivity/specificity"""
    
    # Load model
    model = load_model(model_path)
    if model is None:
        print("Failed to load model")
        return None
    
    # Load annotations
    annotations_file = os.path.join(dataset_path, 'annotations.csv')
    if not os.path.exists(annotations_file):
        print(f"Annotations file not found: {annotations_file}")
        return None
    
    ground_truth = load_annotations(annotations_file)
    print(f"Loaded annotations for {len(ground_truth)} images")
    
    # Initialize counters
    stats = {
        'true_positives': 0,
        'false_positives': 0,
        'false_negatives': 0,
        'true_negatives': 0,
        'total_images': 0,
        'images_with_lesions': 0,
        'images_without_lesions': 0
    }
    
    class_stats = defaultdict(lambda: {
        'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0
    })
    
    images_dir = os.path.join(dataset_path, 'images')
    
    for image_filename in os.listdir(images_dir):
        if not image_filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
            
        image_path = os.path.join(images_dir, image_filename)
        stats['total_images'] += 1
        
        # Get relative path as it appears in annotations
        relative_path = f"images/{image_filename}"
        gt_annotations = ground_truth.get(relative_path, [])
        
        if len(gt_annotations) > 0:
            stats['images_with_lesions'] += 1
        else:
            stats['images_without_lesions'] += 1
        
        print(f"Processing {image_filename} ({stats['total_images']}) - GT: {len(gt_annotations)} lesions")
        
        try:
            # Preprocess image
            image_tensor, scale = preprocess_image(image_path)
            
            # Run inference
            detections = run_inference(model, image_tensor, score_threshold)
            
            # Scale detections back to original image size
            for detection in detections:
                detection['bbox'] = detection['bbox'] / scale
            
            # Match detections with ground truth
            matched_gt = set()
            matched_det = set()
            
            # For each detection, find best matching ground truth
            for det_idx, detection in enumerate(detections):
                best_iou = 0
                best_gt_idx = -1
                
                for gt_idx, gt_annotation in enumerate(gt_annotations):
                    if gt_idx in matched_gt:
                        continue
                    
                    iou = calculate_iou(detection['bbox'], gt_annotation['bbox'])
                    if iou > best_iou and iou >= iou_threshold:
                        # Check if classes match
                        if detection['class'] == gt_annotation['class']:
                            best_iou = iou
                            best_gt_idx = gt_idx
                
                if best_gt_idx >= 0:
                    # True positive
                    stats['true_positives'] += 1
                    class_stats[detection['class']]['tp'] += 1
                    matched_gt.add(best_gt_idx)
                    matched_det.add(det_idx)
                else:
                    # False positive
                    stats['false_positives'] += 1
                    class_stats[detection['class']]['fp'] += 1
            
            # Count false negatives (unmatched ground truth)
            for gt_idx, gt_annotation in enumerate(gt_annotations):
                if gt_idx not in matched_gt:
                    stats['false_negatives'] += 1
                    class_stats[gt_annotation['class']]['fn'] += 1
            
            # For images without lesions, if no detections -> true negative
            if len(gt_annotations) == 0 and len(detections) == 0:
                stats['true_negatives'] += 1
            elif len(gt_annotations) == 0 and len(detections) > 0:
                # False positives already counted above
                pass
                
        except Exception as e:
            print(f"Error processing {image_filename}: {e}")
            continue
    
    # Calculate metrics
    results = calculate_metrics(stats, class_stats)
    return results


def calculate_metrics(stats, class_stats):
    """Calculate sensitivity, specificity, and other metrics"""
    
    # Overall metrics
    tp = stats['true_positives']
    fp = stats['false_positives'] 
    fn = stats['false_negatives']
    tn = stats['true_negatives']
    
    # Sensitivity (True Positive Rate) = TP / (TP + FN)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # Specificity (True Negative Rate) = TN / (TN + FP)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # Precision = TP / (TP + FP)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    # F1 Score = 2 * (Precision * Sensitivity) / (Precision + Sensitivity)
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0
    
    results = {
        'overall': {
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'f1_score': f1_score,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'true_negatives': tn,
            'total_images': stats['total_images'],
            'images_with_lesions': stats['images_with_lesions'],
            'images_without_lesions': stats['images_without_lesions']
        },
        'per_class': {}
    }
    
    # Per-class metrics
    for class_name, class_stat in class_stats.items():
        tp_c = class_stat['tp']
        fp_c = class_stat['fp']
        fn_c = class_stat['fn']
        
        sens_c = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0.0
        prec_c = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0.0
        f1_c = 2 * (prec_c * sens_c) / (prec_c + sens_c) if (prec_c + sens_c) > 0 else 0.0
        
        results['per_class'][class_name] = {
            'sensitivity': sens_c,
            'precision': prec_c,
            'f1_score': f1_c,
            'true_positives': tp_c,
            'false_positives': fp_c,
            'false_negatives': fn_c
        }
    
    return results


def print_results(results):
    """Print evaluation results in a formatted way"""
    overall = results['overall']
    
    print("\n" + "="*80)
    print("MAMMOGRAM DETECTION PERFORMANCE EVALUATION")
    print("="*80)
    
    print(f"\nDataset Summary:")
    print(f"  Total images processed: {overall['total_images']}")
    print(f"  Images with lesions: {overall['images_with_lesions']}")
    print(f"  Images without lesions: {overall['images_without_lesions']}")
    
    print(f"\nOverall Performance:")
    print(f"  Sensitivity (Recall): {overall['sensitivity']:.3f} ({overall['sensitivity']*100:.1f}%)")
    print(f"  Specificity: {overall['specificity']:.3f} ({overall['specificity']*100:.1f}%)")
    print(f"  Precision: {overall['precision']:.3f} ({overall['precision']*100:.1f}%)")
    print(f"  F1-Score: {overall['f1_score']:.3f}")
    
    print(f"\nConfusion Matrix:")
    print(f"  True Positives (TP): {overall['true_positives']}")
    print(f"  False Positives (FP): {overall['false_positives']}")
    print(f"  False Negatives (FN): {overall['false_negatives']}")
    print(f"  True Negatives (TN): {overall['true_negatives']}")
    
    print(f"\nPer-Class Performance:")
    print("-" * 80)
    for class_name, metrics in results['per_class'].items():
        print(f"\n{class_name.upper()}:")
        print(f"  Sensitivity: {metrics['sensitivity']:.3f} ({metrics['sensitivity']*100:.1f}%)")
        print(f"  Precision: {metrics['precision']:.3f} ({metrics['precision']*100:.1f}%)")
        print(f"  F1-Score: {metrics['f1_score']:.3f}")
        print(f"  TP: {metrics['true_positives']}, FP: {metrics['false_positives']}, FN: {metrics['false_negatives']}")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description='Calculate sensitivity and specificity for mammogram detection model')
    
    parser.add_argument('--dataset_path', required=True, help='Path to dataset directory (containing images/ and annotations.csv)')
    parser.add_argument('--model_path', required=True, help='Path to trained model checkpoint')
    parser.add_argument('--score_threshold', type=float, default=0.1, help='Score threshold for detections (default: 0.1)')
    parser.add_argument('--iou_threshold', type=float, default=0.5, help='IoU threshold for matching detections to ground truth (default: 0.5)')
    parser.add_argument('--output_file', help='Optional: Save results to CSV file')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.dataset_path):
        print(f"Error: Dataset path not found: {args.dataset_path}")
        return
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model path not found: {args.model_path}")
        return
    
    annotations_file = os.path.join(args.dataset_path, 'annotations.csv')
    if not os.path.exists(annotations_file):
        print(f"Error: Annotations file not found: {annotations_file}")
        return
    
    images_dir = os.path.join(args.dataset_path, 'images')
    if not os.path.exists(images_dir):
        print(f"Error: Images directory not found: {images_dir}")
        return
    
    print(f"Starting evaluation...")
    print(f"Dataset: {args.dataset_path}")
    print(f"Model: {args.model_path}")
    print(f"Score threshold: {args.score_threshold}")
    print(f"IoU threshold: {args.iou_threshold}")
    
    # Run evaluation
    results = evaluate_performance(
        args.dataset_path, 
        args.model_path, 
        args.score_threshold, 
        args.iou_threshold
    )
    
    if results is None:
        print("Evaluation failed")
        return
    
    # Print results
    print_results(results)
    
    # Save to file if requested
    if args.output_file:
        save_results_to_csv(results, args.output_file, args)
        print(f"\nResults saved to: {args.output_file}")


def save_results_to_csv(results, output_file, args):
    """Save evaluation results to CSV file"""
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow(['Evaluation Results'])
        writer.writerow(['Dataset', args.dataset_path])
        writer.writerow(['Model', args.model_path])
        writer.writerow(['Score Threshold', args.score_threshold])
        writer.writerow(['IoU Threshold', args.iou_threshold])
        writer.writerow([])
        
        # Overall metrics
        overall = results['overall']
        writer.writerow(['Overall Performance'])
        writer.writerow(['Metric', 'Value', 'Percentage'])
        writer.writerow(['Sensitivity', f"{overall['sensitivity']:.3f}", f"{overall['sensitivity']*100:.1f}%"])
        writer.writerow(['Specificity', f"{overall['specificity']:.3f}", f"{overall['specificity']*100:.1f}%"])
        writer.writerow(['Precision', f"{overall['precision']:.3f}", f"{overall['precision']*100:.1f}%"])
        writer.writerow(['F1-Score', f"{overall['f1_score']:.3f}", ''])
        writer.writerow([])
        
        # Per-class metrics
        writer.writerow(['Per-Class Performance'])
        writer.writerow(['Class', 'Sensitivity', 'Precision', 'F1-Score', 'TP', 'FP', 'FN'])
        for class_name, metrics in results['per_class'].items():
            writer.writerow([
                class_name,
                f"{metrics['sensitivity']:.3f}",
                f"{metrics['precision']:.3f}",
                f"{metrics['f1_score']:.3f}",
                metrics['true_positives'],
                metrics['false_positives'],
                metrics['false_negatives']
            ])


if __name__ == '__main__':
    main()