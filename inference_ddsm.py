import argparse
import torch
import numpy as np
import cv2
import os
import sys
from retinanet import model
from retinanet.dataloader import Normalizer, Resizer
from torchvision import transforms
import time

# DDSM dataset classes for mammogram mass detection
DDSM_CLASSES = {
    0: 'benign',
    1: 'malignant'
}

def load_model(model_path):
    """Load RetinaNet model from checkpoint"""
    try:
        # Try loading the full model
        print(f"Attempting to load model from {model_path}")
        if torch.cuda.is_available():
            retinanet = torch.load(model_path, map_location=torch.device('cuda'))
        else:
            retinanet = torch.load(model_path, map_location=torch.device('cpu'))
            
        # If it's wrapped in DataParallel, get the module
        if isinstance(retinanet, torch.nn.DataParallel):
            retinanet = retinanet.module
    except Exception as e:
        print(f"Error loading model as full model: {e}")
        
        try:
            # Create model with 2 classes and try loading state dict
            print("Trying to load as state dict...")
            retinanet = model.resnet50(num_classes=3)
            
            # Try to load the model weights
            if torch.cuda.is_available():
                state_dict = torch.load(model_path, map_location=torch.device('cuda'))
                retinanet.load_state_dict(state_dict)
                retinanet = retinanet.cuda()
            else:
                state_dict = torch.load(model_path, map_location=torch.device('cpu'))
                retinanet.load_state_dict(state_dict)
        except Exception as e2:
            print(f"Error loading model as state dict: {e2}")
            raise RuntimeError(f"Failed to load model: {e2}")
    
    # Set model to evaluation mode
    retinanet.training = False
    retinanet.eval()
    return retinanet

def preprocess_image(image_path):
    """Preprocess mammogram image for inference"""
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at path: {image_path}")
    
    # Make a copy of the original image for drawing
    original_image = image.copy()
    
    # For mammograms, they might be grayscale; ensure proper processing
    if len(image.shape) == 2:
        # If grayscale, convert to 3 channels
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        # If already RGB/BGR, convert BGR to RGB (since model expects RGB)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Get image dimensions
    rows, cols, cns = image.shape
    
    # Calculate scale for resizing
    smallest_side = min(rows, cols)
    min_side = 800  # Larger size for mammograms to preserve details
    max_side = 1333
    scale = min_side / smallest_side
    
    # Check if the largest side exceeds max_side
    largest_side = max(rows, cols)
    if largest_side * scale > max_side:
        scale = max_side / largest_side
    
    # Resize the image
    image = cv2.resize(image, (int(round(cols * scale)), int(round((rows * scale)))))
    rows, cols, cns = image.shape
    
    # Pad to be divisible by 32
    pad_w = 32 - rows % 32
    pad_h = 32 - cols % 32
    
    new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
    new_image[:rows, :cols, :] = image.astype(np.float32)
    image = new_image.astype(np.float32)
    
    # Normalize image
    image /= 255.0
    image -= [0.485, 0.456, 0.406]
    image /= [0.229, 0.224, 0.225]
    
    # Convert to tensor and reshape for model input
    image = np.expand_dims(image, 0)
    image = np.transpose(image, (0, 3, 1, 2))
    image = torch.from_numpy(image)
    
    return original_image, image, scale

def draw_caption(image, box, caption, color=(0, 0, 255)):
    """Draw a caption above the box in an image"""
    b = np.array(box).astype(int)
    
    # Background rectangle for text
    cv2.rectangle(image, (b[0], b[1] - 30), (b[0] + len(caption) * 10, b[1]), (0, 0, 0), -1)
    
    # Text
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

def get_prediction_color(class_name):
    """Return different colors based on the prediction class"""
    if class_name == 'benign':
        return (0, 255, 0)  # Green for benign
    else:
        return (0, 0, 255)  # Red for malignant

def main():
    parser = argparse.ArgumentParser(description='RetinaNet DDSM mammogram inference script')
    parser.add_argument('--image', help='Path to input mammogram image', required=True)
    parser.add_argument('--model', help='Path to model checkpoint', default='model_final.pt')
    parser.add_argument('--score-threshold', type=float, help='Score threshold for detections', default=0.1)
    parser.add_argument('--output', help='Path to save output image', default='output_ddsm.jpg')
    parser.add_argument('--num-classes', type=int, help='Number of classes in the model', default=2)
    
    args = parser.parse_args()
    
    # Check if the image exists
    if not os.path.isfile(args.image):
        print(f"Error: Image file '{args.image}' not found.")
        sys.exit(1)
    
    # Check if the model exists
    if not os.path.isfile(args.model):
        print(f"Error: Model file '{args.model}' not found.")
        sys.exit(1)
    
    # Load model
    print("Loading model...")
    try:
        retinanet = load_model(args.model)
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)
    
    # Load and preprocess image
    print(f"Processing mammogram image: {args.image}")
    try:
        original_image, image_tensor, scale = preprocess_image(args.image)
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        sys.exit(1)
    
    # Run inference
    with torch.no_grad():
        if torch.cuda.is_available():
            image_tensor = image_tensor.cuda()
        
        start_time = time.time()
        try:
            scores, labels, boxes = retinanet(image_tensor.float())
            print(f"Inference time: {time.time() - start_time:.4f} seconds")
        except Exception as e:
            print(f"Error during inference: {e}")
            sys.exit(1)
    
    # Filter detections by score threshold
    idxs = np.where(scores.cpu().numpy() > args.score_threshold)[0]
    
    # Draw bounding boxes and labels
    print(f"Found {len(idxs)} detections above threshold {args.score_threshold}")
    
    # Store results for later reporting
    results = []
    benign_count = 0
    malignant_count = 0
    
    # Process and draw detections
    for i in idxs:
        score = scores[i].item()
        label = int(labels[i].item())
        box = boxes[i, :].cpu().numpy() / scale
        
        # Convert box coordinates to integers
        box_coords = [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
        
        # Get class name
        class_name = DDSM_CLASSES.get(label, f"Class {label}")
        color = get_prediction_color(class_name)
        caption = f"{class_name}: {score:.3f}"
        
        # Store result
        results.append({
            'class': class_name,
            'score': score,
            'box': box_coords
        })
        
        # Count by class
        if class_name == 'benign':
            benign_count += 1
        elif class_name == 'malignant':
            malignant_count += 1
        
        # Draw bounding box
        cv2.rectangle(original_image, (box_coords[0], box_coords[1]), (box_coords[2], box_coords[3]), color, 2)
        
        # Draw caption
        draw_caption(original_image, box_coords, caption, color)
    
    # Summarize findings with largest mass or highest confidence finding
    if results:
        # Sort by confidence score
        results.sort(key=lambda x: x['score'], reverse=True)
        highest_conf = results[0]
        
        # Calculate mass area for each detection
        for result in results:
            box = result['box']
            width = box[2] - box[0]
            height = box[3] - box[1]
            result['area'] = width * height
        
        # Find largest mass
        largest_mass = max(results, key=lambda x: x['area'])
        
        # Draw summary on the image
        summary_text = []
        summary_text.append(f"Highest confidence: {highest_conf['class']} ({highest_conf['score']:.3f})")
        
        if largest_mass != highest_conf:
            summary_text.append(f"Largest mass: {largest_mass['class']} ({largest_mass['score']:.3f})")
        
        summary_text.append(f"Summary: {benign_count} benign, {malignant_count} malignant masses detected")
        
        # Add summary text to the top of the image
        for i, text in enumerate(summary_text):
            y_pos = 30 + i * 30
            cv2.putText(original_image, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        # If no detections, add text indicating no findings
        cv2.putText(original_image, "No significant findings detected", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Save the output image
    print(f"Saving result to: {args.output}")
    cv2.imwrite(args.output, original_image)
    
    # Print summary to console
    print("\n===== MAMMOGRAM ANALYSIS RESULTS =====")
    if results:
        print(f"Total detections: {len(results)}")
        print(f"Benign masses: {benign_count}")
        print(f"Malignant masses: {malignant_count}")
        print(f"Highest confidence detection: {highest_conf['class']} with {highest_conf['score']:.3f} confidence")
    else:
        print("No significant findings detected")
    print("=====================================")

if __name__ == '__main__':
    main()
