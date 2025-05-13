import argparse
import torch
import numpy as np
import cv2
import os
from retinanet import model
from retinanet.dataloader import Normalizer, Resizer
from torchvision import transforms
import time

# COCO dataset classes
COCO_CLASSES = {
    0: 'benign', 
    1: 'malignant'
}

def load_model(model_path):
    """Load RetinaNet model from checkpoint"""
    # Create model with 80 classes (COCO has 80 classes)
    retinanet = model.resnet50(num_classes=2)
    
    # Load model weights
    if torch.cuda.is_available():
        retinanet.load_state_dict(torch.load(model_path, weights_only=False))
        retinanet = retinanet.cuda()
    else:
        retinanet.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'),weights_only=False))
    
    # Set model to evaluation mode
    retinanet.training = False
    retinanet.eval()
    return retinanet

def preprocess_image(image_path):
    """Preprocess image for inference"""
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at path: {image_path}")
    
    # Make a copy of the original image for drawing
    original_image = image.copy()
    
    # Convert BGR to RGB (since model expects RGB)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Get image dimensions
    rows, cols, cns = image.shape
    
    # Calculate scale for resizing
    smallest_side = min(rows, cols)
    min_side = 608
    max_side = 1024
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

def draw_caption(image, box, caption):
    """Draw a caption above the box in an image"""
    b = np.array(box).astype(int)
    
    # Background rectangle for text
    cv2.rectangle(image, (b[0], b[1] - 30), (b[0] + len(caption) * 10, b[1]), (0, 0, 0), -1)
    
    # Text
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

def main():
    parser = argparse.ArgumentParser(description='RetinaNet single image inference script')
    parser.add_argument('--image', help='Path to input image', required=True)
    parser.add_argument('--model', help='Path to model checkpoint', default='model_final.pt')
    parser.add_argument('--score-threshold', type=float, help='Score threshold for detections', default=0.5)
    parser.add_argument('--output', help='Path to save output image', default='output.jpg')
    
    args = parser.parse_args()
    
    # Load model
    print("Loading model...")
    retinanet = load_model(args.model)
    
    # Load and preprocess image
    print(f"Processing image: {args.image}")
    original_image, image_tensor, scale = preprocess_image(args.image)
    
    # Run inference
    with torch.no_grad():
        if torch.cuda.is_available():
            image_tensor = image_tensor.cuda()
        
        start_time = time.time()
        scores, labels, boxes = retinanet(image_tensor.float())
        print(f"Inference time: {time.time() - start_time:.4f} seconds")
    
    # Filter detections by score threshold
    idxs = np.where(scores.cpu().numpy() > args.score_threshold)[0]
    
    # Draw bounding boxes and labels
    print(f"Found {len(idxs)} detections above threshold {args.score_threshold}")
    
    # Process and draw detections
    for i in idxs:
        score = scores[i].item()
        label = int(labels[i].item())
        box = boxes[i, :].cpu().numpy() / scale
        
        # Convert box coordinates to integers
        box_coords = [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
        
        # Get class name
        class_name = COCO_CLASSES.get(label, f"Class {label}")
        caption = f"{class_name}: {score:.2f}"
        
        # Draw bounding box
        cv2.rectangle(original_image, (box_coords[0], box_coords[1]), (box_coords[2], box_coords[3]), (0, 0, 255), 2)
        
        # Draw caption
        draw_caption(original_image, box_coords, caption)
    
    # Save the output image
    print(f"Saving result to: {args.output}")
    cv2.imwrite(args.output, original_image)
    print(f"Done! Detected {len(idxs)} objects.")

if __name__ == '__main__':
    main()
