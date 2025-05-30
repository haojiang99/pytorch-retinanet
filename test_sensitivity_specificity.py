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

# Class mapping for DDSM dataset - this might need adjustment based on your model
DDSM_CLASSES = {
    0: 'benign mass', 
    1: 'malignant mass',
    2: 'benign calcification',
    3: 'malignant calcification'
}

# Alternative mappings in case the model was trained differently
ALTERNATIVE_MAPPINGS = [
    # If model has only 2 classes (benign vs malignant)
    {0: 'benign mass', 1: 'malignant mass'},
    # If model uses different order
    {0: 'malignant mass', 1: 'benign mass', 2: 'malignant calcification', 3: 'benign calcification'},
    # Single class models
    {0: 'mass'},
]


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
    
    # Debug: Check model's number of classes
    try:
        if hasattr(retinanet, 'classificationModel'):
            num_classes = retinanet.classificationModel.num_classes
            print(f"Model expects {num_classes} classes")
        elif hasattr(retinanet, 'module') and hasattr(retinanet.module, 'classificationModel'):
            num_classes = retinanet.module.classificationModel.num_classes
            print(f"Model expects {num_classes} classes")
    except:
        print("Could not determine model's expected number of classes")
    
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


def load_class_mapping(dataset_path):
    """Load class mapping from class_map.csv if available"""
    class_map_file = os.path.join(dataset_path, 'class_map.csv')
    class_mapping = {}
    
    if os.path.exists(class_map_file):
        print(f"Loading class mapping from {class_map_file}")
        with open(class_map_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    class_name = row[0].strip()
                    class_id = int(row[1])
                    class_mapping[class_id] = class_name
        
        print(f"Found class mapping: {class_mapping}")
        return class_mapping
    else:
        print(f"No class_map.csv found, using default DDSM mapping")
        return DDSM_CLASSES


def load_annotations(annotations_file):
    """Load ground truth annotations from CSV file - focus on lesion presence only"""
    image_lesions = defaultdict(set)  # Store unique lesion types per image
    unique_classes = set()  # Track all unique class names found
    
    with open(annotations_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 6:  # path,x1,y1,x2,y2,class_name
                image_path = row[0]
                class_name = row[5].strip() if len(row) > 5 else ''
                
                # Track all unique class names for debugging
                if class_name != '':
                    unique_classes.add(class_name)
                
                # Skip empty annotations (negative examples)
                if class_name == '' or row[1] == '' or row[2] == '' or row[3] == '' or row[4] == '':
                    # This is a negative example - no lesions
                    if image_path not in image_lesions:
                        image_lesions[image_path] = set()
                    continue
                
                # Valid lesion annotation - normalize class name
                class_name_lower = class_name.lower().strip()
                
                # Map variations to standard names
                if 'benign' in class_name_lower and 'mass' in class_name_lower:
                    standard_name = 'benign mass'
                elif 'malignant' in class_name_lower and 'mass' in class_name_lower:
                    standard_name = 'malignant mass'
                elif 'benign' in class_name_lower and 'calc' in class_name_lower:
                    standard_name = 'benign calcification'
                elif 'malignant' in class_name_lower and 'calc' in class_name_lower:
                    standard_name = 'malignant calcification'
                else:
                    print(f"Warning: Unknown class name '{class_name}' in annotations")
                    continue
                
                image_lesions[image_path].add(standard_name)
    
    print(f"Found unique class names in annotations: {sorted(unique_classes)}")
    print(f"Loaded {len(image_lesions)} images with annotations")
    
    # Print summary of lesion types
    lesion_counts = defaultdict(int)
    for image_path, lesions in image_lesions.items():
        for lesion in lesions:
            lesion_counts[lesion] += 1
    
    print("Ground truth lesion distribution:")
    for lesion_type, count in sorted(lesion_counts.items()):
        print(f"  {lesion_type}: {count} images")
    
    return image_lesions




def run_inference(model, image_tensor, score_threshold=0.1):
    """Run inference on a single image - return detected lesion types only"""
    with torch.no_grad():
        if torch.cuda.is_available():
            image_tensor = image_tensor.cuda()
        
        try:
            scores, labels, boxes = model(image_tensor)
            
            # Debug: print all detections above a very low threshold
            debug_detections = []
            for i in range(len(scores)):
                if scores[i] > 0.01:  # Very low threshold for debugging
                    class_id = int(labels[i].cpu().numpy())
                    class_name = DDSM_CLASSES.get(class_id, f'unknown_class_{class_id}')
                    score = scores[i].cpu().numpy()
                    debug_detections.append((class_name, score, class_id))
            
            # Filter by score threshold and collect unique lesion types
            detected_lesions = set()
            valid_detections = []
            for i in range(len(scores)):
                if scores[i] > score_threshold:
                    class_id = int(labels[i].cpu().numpy())
                    class_name = DDSM_CLASSES.get(class_id, f'unknown_class_{class_id}')
                    score = scores[i].cpu().numpy()
                    valid_detections.append((class_name, score, class_id))
                    
                    if class_name in ['benign mass', 'malignant mass', 'benign calcification', 'malignant calcification']:
                        detected_lesions.add(class_name)
            
            # Debug output (controlled by global debug flag)
            if hasattr(run_inference, 'debug_mode') and run_inference.debug_mode:
                if debug_detections:
                    print(f"    All detections > 0.01: {debug_detections[:5]}")  # Show first 5
                if valid_detections:
                    print(f"    Valid detections > {score_threshold}: {valid_detections}")
                elif len(debug_detections) == 0:
                    print(f"    No detections found at all")
            
            return detected_lesions
            
        except Exception as e:
            print(f"Error during inference: {e}")
            return set()


def evaluate_performance(dataset_path, model_path, score_threshold=0.1):
    """Evaluate model performance and calculate sensitivity/specificity based on lesion presence"""
    
    # Load class mapping
    global DDSM_CLASSES
    DDSM_CLASSES = load_class_mapping(dataset_path)
    
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
    
    # Initialize counters for overall and per-class metrics
    overall_stats = {
        'true_positives': 0,    # Images with lesions correctly identified as having lesions
        'false_positives': 0,   # Images without lesions incorrectly identified as having lesions
        'false_negatives': 0,   # Images with lesions incorrectly identified as not having lesions
        'true_negatives': 0,    # Images without lesions correctly identified as not having lesions
        'total_images': 0,
        'images_with_lesions': 0,
        'images_without_lesions': 0
    }
    
    # Per-class statistics (presence/absence of each lesion type)
    class_stats = {}
    for lesion_type in ['benign mass', 'malignant mass', 'benign calcification', 'malignant calcification']:
        class_stats[lesion_type] = {
            'tp': 0,  # Images with this lesion type correctly detected
            'fp': 0,  # Images without this lesion type but incorrectly detected
            'fn': 0,  # Images with this lesion type but not detected
            'tn': 0   # Images without this lesion type correctly not detected
        }
    
    images_dir = os.path.join(dataset_path, 'images')
    
    for image_filename in os.listdir(images_dir):
        if not image_filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
            
        image_path = os.path.join(images_dir, image_filename)
        overall_stats['total_images'] += 1
        
        # Get relative path as it appears in annotations
        relative_path = f"images/{image_filename}"
        gt_lesions = ground_truth.get(relative_path, set())
        
        has_lesions = len(gt_lesions) > 0
        if has_lesions:
            overall_stats['images_with_lesions'] += 1
        else:
            overall_stats['images_without_lesions'] += 1
        
        print(f"Processing {image_filename} ({overall_stats['total_images']}) - GT lesions: {gt_lesions}")
        
        try:
            # Preprocess image
            image_tensor, scale = preprocess_image(image_path)
            
            # Run inference - get detected lesion types
            detected_lesions = run_inference(model, image_tensor, score_threshold)
            
            print(f"  Detected lesions: {detected_lesions}")
            
            # Overall performance: any lesion vs no lesion
            detected_any = len(detected_lesions) > 0
            
            if has_lesions and detected_any:
                overall_stats['true_positives'] += 1
            elif has_lesions and not detected_any:
                overall_stats['false_negatives'] += 1
            elif not has_lesions and detected_any:
                overall_stats['false_positives'] += 1
            elif not has_lesions and not detected_any:
                overall_stats['true_negatives'] += 1
            
            # Per-class performance: presence/absence of each specific lesion type
            for lesion_type in class_stats.keys():
                gt_has_lesion = lesion_type in gt_lesions
                detected_lesion = lesion_type in detected_lesions
                
                if gt_has_lesion and detected_lesion:
                    class_stats[lesion_type]['tp'] += 1
                elif gt_has_lesion and not detected_lesion:
                    class_stats[lesion_type]['fn'] += 1
                elif not gt_has_lesion and detected_lesion:
                    class_stats[lesion_type]['fp'] += 1
                elif not gt_has_lesion and not detected_lesion:
                    class_stats[lesion_type]['tn'] += 1
                
        except Exception as e:
            print(f"Error processing {image_filename}: {e}")
            continue
    
    # Calculate metrics
    results = calculate_metrics(overall_stats, class_stats)
    return results


def calculate_metrics(overall_stats, class_stats):
    """Calculate sensitivity, specificity, and other metrics"""
    
    # Overall metrics (any lesion vs no lesion)
    tp = overall_stats['true_positives']
    fp = overall_stats['false_positives'] 
    fn = overall_stats['false_negatives']
    tn = overall_stats['true_negatives']
    
    # Sensitivity (True Positive Rate) = TP / (TP + FN)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # Specificity (True Negative Rate) = TN / (TN + FP)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # Precision (Positive Predictive Value) = TP / (TP + FP)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    # Negative Predictive Value = TN / (TN + FN)
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    
    # Accuracy = (TP + TN) / (TP + TN + FP + FN)
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    
    # F1 Score = 2 * (Precision * Sensitivity) / (Precision + Sensitivity)
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0
    
    results = {
        'overall': {
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'negative_predictive_value': npv,
            'accuracy': accuracy,
            'f1_score': f1_score,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'true_negatives': tn,
            'total_images': overall_stats['total_images'],
            'images_with_lesions': overall_stats['images_with_lesions'],
            'images_without_lesions': overall_stats['images_without_lesions']
        },
        'per_class': {}
    }
    
    # Per-class metrics (presence/absence of each specific lesion type)
    for class_name, class_stat in class_stats.items():
        tp_c = class_stat['tp']
        fp_c = class_stat['fp']
        fn_c = class_stat['fn']
        tn_c = class_stat['tn']
        
        # Sensitivity for this lesion type
        sens_c = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0.0
        
        # Specificity for this lesion type
        spec_c = tn_c / (tn_c + fp_c) if (tn_c + fp_c) > 0 else 0.0
        
        # Precision for this lesion type
        prec_c = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0.0
        
        # NPV for this lesion type
        npv_c = tn_c / (tn_c + fn_c) if (tn_c + fn_c) > 0 else 0.0
        
        # Accuracy for this lesion type
        acc_c = (tp_c + tn_c) / (tp_c + tn_c + fp_c + fn_c) if (tp_c + tn_c + fp_c + fn_c) > 0 else 0.0
        
        # F1 score for this lesion type
        f1_c = 2 * (prec_c * sens_c) / (prec_c + sens_c) if (prec_c + sens_c) > 0 else 0.0
        
        results['per_class'][class_name] = {
            'sensitivity': sens_c,
            'specificity': spec_c,
            'precision': prec_c,
            'negative_predictive_value': npv_c,
            'accuracy': acc_c,
            'f1_score': f1_c,
            'true_positives': tp_c,
            'false_positives': fp_c,
            'false_negatives': fn_c,
            'true_negatives': tn_c
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
    
    print(f"\nOverall Performance (Any Lesion vs No Lesion):")
    print(f"  Sensitivity (Recall): {overall['sensitivity']:.3f} ({overall['sensitivity']*100:.1f}%)")
    print(f"  Specificity: {overall['specificity']:.3f} ({overall['specificity']*100:.1f}%)")
    print(f"  Precision (PPV): {overall['precision']:.3f} ({overall['precision']*100:.1f}%)")
    print(f"  Negative Predictive Value (NPV): {overall['negative_predictive_value']:.3f} ({overall['negative_predictive_value']*100:.1f}%)")
    print(f"  Accuracy: {overall['accuracy']:.3f} ({overall['accuracy']*100:.1f}%)")
    print(f"  F1-Score: {overall['f1_score']:.3f}")
    
    print(f"\nOverall Confusion Matrix:")
    print(f"  True Positives (TP): {overall['true_positives']} (images with lesions correctly identified)")
    print(f"  False Positives (FP): {overall['false_positives']} (images without lesions incorrectly flagged)")
    print(f"  False Negatives (FN): {overall['false_negatives']} (images with lesions missed)")
    print(f"  True Negatives (TN): {overall['true_negatives']} (images without lesions correctly identified)")
    
    print(f"\nPer-Lesion Type Performance:")
    print("-" * 80)
    for class_name, metrics in results['per_class'].items():
        print(f"\n{class_name.upper()}:")
        print(f"  Sensitivity: {metrics['sensitivity']:.3f} ({metrics['sensitivity']*100:.1f}%) - ability to detect this lesion type")
        print(f"  Specificity: {metrics['specificity']:.3f} ({metrics['specificity']*100:.1f}%) - ability to correctly rule out this lesion type")
        print(f"  Precision (PPV): {metrics['precision']:.3f} ({metrics['precision']*100:.1f}%) - when detected, probability it's correct")
        print(f"  NPV: {metrics['negative_predictive_value']:.3f} ({metrics['negative_predictive_value']*100:.1f}%) - when not detected, probability it's absent")
        print(f"  Accuracy: {metrics['accuracy']:.3f} ({metrics['accuracy']*100:.1f}%)")
        print(f"  F1-Score: {metrics['f1_score']:.3f}")
        print(f"  TP: {metrics['true_positives']}, FP: {metrics['false_positives']}, FN: {metrics['false_negatives']}, TN: {metrics['true_negatives']}")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description='Calculate sensitivity and specificity for mammogram detection model')
    
    parser.add_argument('--dataset_path', required=True, help='Path to dataset directory (containing images/ and annotations.csv)')
    parser.add_argument('--model_path', required=True, help='Path to trained model checkpoint')
    parser.add_argument('--score_threshold', type=float, default=0.1, help='Score threshold for detections (default: 0.1)')
    parser.add_argument('--output_file', help='Optional: Save results to CSV file')
    parser.add_argument('--debug', action='store_true', help='Enable debug output for troubleshooting')
    
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
    
    # Set debug mode
    if args.debug:
        run_inference.debug_mode = True
    
    # Run evaluation
    results = evaluate_performance(
        args.dataset_path, 
        args.model_path, 
        args.score_threshold
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
        writer.writerow([])
        
        # Overall metrics
        overall = results['overall']
        writer.writerow(['Overall Performance (Any Lesion vs No Lesion)'])
        writer.writerow(['Metric', 'Value', 'Percentage'])
        writer.writerow(['Sensitivity', f"{overall['sensitivity']:.3f}", f"{overall['sensitivity']*100:.1f}%"])
        writer.writerow(['Specificity', f"{overall['specificity']:.3f}", f"{overall['specificity']*100:.1f}%"])
        writer.writerow(['Precision', f"{overall['precision']:.3f}", f"{overall['precision']*100:.1f}%"])
        writer.writerow(['NPV', f"{overall['negative_predictive_value']:.3f}", f"{overall['negative_predictive_value']*100:.1f}%"])
        writer.writerow(['Accuracy', f"{overall['accuracy']:.3f}", f"{overall['accuracy']*100:.1f}%"])
        writer.writerow(['F1-Score', f"{overall['f1_score']:.3f}", ''])
        writer.writerow([])
        
        # Per-class metrics
        writer.writerow(['Per-Lesion Type Performance'])
        writer.writerow(['Class', 'Sensitivity', 'Specificity', 'Precision', 'NPV', 'Accuracy', 'F1-Score', 'TP', 'FP', 'FN', 'TN'])
        for class_name, metrics in results['per_class'].items():
            writer.writerow([
                class_name,
                f"{metrics['sensitivity']:.3f}",
                f"{metrics['specificity']:.3f}",
                f"{metrics['precision']:.3f}",
                f"{metrics['negative_predictive_value']:.3f}",
                f"{metrics['accuracy']:.3f}",
                f"{metrics['f1_score']:.3f}",
                metrics['true_positives'],
                metrics['false_positives'],
                metrics['false_negatives'],
                metrics['true_negatives']
            ])


if __name__ == '__main__':
    main()