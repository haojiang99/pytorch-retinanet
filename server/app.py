import os
import uuid
import cv2
import torch
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import sys
import time
import base64
import logging

# Import configuration
from config import MODEL_PATH, SERVER_HOST, SERVER_PORT, DEBUG_MODE

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path to import retinanet modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from retinanet import model
from retinanet.model import ResNet, Bottleneck, BasicBlock
from torch import serialization

# Configure Flask app
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})  # Enable Cross-Origin Resource Sharing for /api routes

# Request logging
@app.before_request
def log_request_info():
    logger.debug('Request Headers: %s', request.headers)
    logger.debug('Request Method: %s', request.method)
    logger.debug('Request URL: %s', request.url)
    logger.debug('Request Path: %s', request.path)
    if request.method == 'POST':
        logger.debug('Form Data: %s', request.form)
        logger.debug('Files: %s', [f for f in request.files])

# Create upload directory for temporary storage of images
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
RESULT_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# DDSM dataset classes for mammogram detection
DDSM_CLASSES = {
    0: 'benign mass', 
    1: 'malignant mass',
    2: 'benign calcification',
    3: 'malignant calcification'
}

# Load model
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ddsm_checkpoints', 'ddsm_retinanet_final.pt')
print(MODEL_PATH)
retinanet = None

def load_model(model_path=MODEL_PATH):
    """Load RetinaNet model from checkpoint"""
    # Add safe globals for PyTorch 2.6+
    serialization.add_safe_globals([ResNet, Bottleneck, BasicBlock])
    
    try:
        # First try to load the entire model with weights_only=False
        print(f"Loading model from: {model_path}")
        if torch.cuda.is_available():
            retinanet = torch.load(model_path, map_location=torch.device('cuda'), weights_only=False)
        else:
            retinanet = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        
        # If it's wrapped in DataParallel, get the module
        if isinstance(retinanet, torch.nn.DataParallel):
            print("Loaded DataParallel model, extracting module...")
            retinanet = retinanet.module
            
    except Exception as e:
        print(f"Error loading full model: {e}")
        print("Trying to create model and load state dict...")
        
        # Try creating a new model and loading state dict with weights_only=False
        retinanet = model.resnet50(num_classes=2)
        
        try:
            if torch.cuda.is_available():
                state_dict = torch.load(model_path, map_location=torch.device('cuda'), weights_only=False)
                if isinstance(state_dict, torch.nn.Module):
                    print("Loaded a full model instead of state dict, extracting state dict...")
                    state_dict = state_dict.state_dict()
                retinanet.load_state_dict(state_dict)
                retinanet = retinanet.cuda()
            else:
                state_dict = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
                if isinstance(state_dict, torch.nn.Module):
                    print("Loaded a full model instead of state dict, extracting state dict...")
                    state_dict = state_dict.state_dict()
                retinanet.load_state_dict(state_dict)
        except Exception as e2:
            print(f"Error loading state dict: {e2}")
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
    if 'benign mass' in class_name:
        return (0, 255, 0)  # Green for benign mass
    elif 'malignant mass' in class_name:
        return (0, 0, 255)  # Red for malignant mass
    elif 'benign calcification' in class_name:
        return (255, 255, 0)  # Yellow for benign calcification
    else:
        return (255, 0, 0)  # Blue for malignant calcification

def process_image(image_path, score_threshold=0.1, use_gemini=True):
    """Process an image and return detection results"""
    global retinanet
    
    # Load model if not loaded
    if retinanet is None:
        retinanet = load_model()
    
    # Preprocess image
    try:
        original_image, image_tensor, scale = preprocess_image(image_path)
    except Exception as e:
        return None, None, str(e)
    
    # Run inference
    with torch.no_grad():
        if torch.cuda.is_available():
            image_tensor = image_tensor.cuda()
        
        try:
            scores, labels, boxes = retinanet(image_tensor.float())
        except Exception as e:
            return None, None, str(e)
    
    # Filter detections by score threshold
    idxs = np.where(scores.cpu().numpy() > score_threshold)[0]
    
    # Store results
    results = []
    mass_benign_count = 0
    mass_malignant_count = 0
    calc_benign_count = 0
    calc_malignant_count = 0
    
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
        if class_name == 'benign mass':
            mass_benign_count += 1
        elif class_name == 'malignant mass':
            mass_malignant_count += 1
        elif class_name == 'benign calcification':
            calc_benign_count += 1
        elif class_name == 'malignant calcification':
            calc_malignant_count += 1
        
        # Draw bounding box
        cv2.rectangle(original_image, (box_coords[0], box_coords[1]), (box_coords[2], box_coords[3]), color, 2)
        
        # Draw caption
        draw_caption(original_image, box_coords, caption, color)
    
    # Summary info
    summary = {
        'total': len(results),
        'mass_benign': mass_benign_count,
        'mass_malignant': mass_malignant_count,
        'calc_benign': calc_benign_count,
        'calc_malignant': calc_malignant_count,
        'findings': []
    }
    
    # Process findings for summary
    if results:
        # Sort by confidence score
        results.sort(key=lambda x: x['score'], reverse=True)
        highest_conf = results[0]
        summary['highest_confidence'] = highest_conf
        
        # Calculate relative size for each detection (for sorting purposes only)
        for result in results:
            box = result['box']
            width = box[2] - box[0]
            height = box[3] - box[1]
            result['area'] = width * height  # Internal use only for size comparison
            
            # Add descriptive size category instead of pixel measurements
            area = width * height
            if area > 10000:
                result['size_description'] = 'large'
            elif area > 5000:
                result['size_description'] = 'moderate'
            else:
                result['size_description'] = 'small'
                
            summary['findings'].append(result)
        
        # Find largest mass
        largest_mass = max(results, key=lambda x: x['area'])
        summary['largest_mass'] = largest_mass
        
        # Draw summary on the image in radiology report style
        summary_text = []
        
        # Primary finding
        summary_text.append(f"Primary finding: {highest_conf['class']} (confidence: {highest_conf['score']:.1%})")
        
        # Additional findings if different from primary
        if largest_mass != highest_conf:
            summary_text.append(f"Additional: {largest_mass['class']} (confidence: {largest_mass['score']:.1%})")
        
        # Summary counts in medical terminology
        if mass_benign_count + mass_malignant_count > 0:
            mass_summary = f"Masses identified: {mass_benign_count} benign, {mass_malignant_count} malignant"
            summary_text.append(mass_summary)
        
        if calc_benign_count + calc_malignant_count > 0:
            calc_summary = f"Calcifications noted: {calc_benign_count} benign, {calc_malignant_count} malignant"
            summary_text.append(calc_summary)
        
        # Add summary text to the top of the image
        for i, text in enumerate(summary_text):
            y_pos = 30 + i * 30
            cv2.putText(original_image, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        # If no detections, add text indicating no findings
        cv2.putText(original_image, "IMPRESSION: No significant abnormalities detected", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Generate unique filename for result
    result_filename = f"{str(uuid.uuid4())}.jpg"
    result_path = os.path.join(RESULT_FOLDER, result_filename)
    
    # Save the output image
    cv2.imwrite(result_path, original_image)
    
    # Create data URI for the result image
    with open(result_path, "rb") as img_file:
        img_data = base64.b64encode(img_file.read()).decode('utf-8')
    
    # Get Gemini analysis if enabled
    gemini_analysis = None
    if use_gemini:
        try:
            # Import Gemini client
            from gemini_client import GeminiClient
            
            # Create Gemini client
            gemini_client = GeminiClient()
            
            # Get Gemini analysis
            gemini_result = gemini_client.analyze_mammogram(original_image, summary)
            
            if gemini_result.get('success', False):
                gemini_analysis = gemini_result.get('gemini_analysis')
                logger.info("Gemini analysis completed successfully")
            else:
                logger.warning(f"Gemini analysis failed: {gemini_result.get('error', 'Unknown error')}")
        except Exception as e:
            logger.error(f"Error during Gemini analysis: {str(e)}")
    
    # Add Gemini analysis to summary
    if gemini_analysis:
        summary['gemini_analysis'] = gemini_analysis
    
    return summary, result_filename, img_data, None

@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for mammogram predictions"""
    logger.info("Received prediction request")
    
    # Check if file was uploaded
    if 'file' not in request.files:
        logger.error("No file in request")
        return jsonify({
            'error': 'No file uploaded'
        }), 400
    
    file = request.files['file']
    logger.info(f"File received: {file.filename}")
    
    # Check if the file has a valid name
    if file.filename == '':
        logger.error("Empty filename")
        return jsonify({
            'error': 'No file selected'
        }), 400
    
    # Check file extension
    allowed_extensions = {'jpg', 'jpeg', 'png'}
    file_ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
    
    if file_ext not in allowed_extensions:
        logger.error(f"Invalid file extension: {file_ext}")
        return jsonify({
            'error': f'Invalid file type. Allowed types: {", ".join(allowed_extensions)}'
        }), 400
    
    # Generate unique filename
    filename = f"{str(uuid.uuid4())}.{file_ext}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    logger.debug(f"Saving file to: {filepath}")
    
    # Save the file
    file.save(filepath)
    
    # Get threshold parameter (optional)
    threshold = float(request.form.get('threshold', 0.1))
    logger.debug(f"Using threshold: {threshold}")
    
    # Get Gemini analysis parameter (optional)
    use_gemini = request.form.get('use_gemini', 'true').lower() == 'true'
    logger.debug(f"Using Gemini analysis: {use_gemini}")
    
    # Process the image
    logger.info("Processing image...")
    summary, result_filename, img_data, error = process_image(filepath, threshold, use_gemini)
    
    if error:
        logger.error(f"Image processing error: {error}")
        return jsonify({
            'error': f'Failed to process image: {error}'
        }), 500
    
    # Clean up the uploaded file
    try:
        os.remove(filepath)
        logger.debug(f"Removed temporary file: {filepath}")
    except Exception as e:
        logger.warning(f"Failed to remove temp file: {e}")
    
    # Create response
    response = {
        'summary': summary,
        'result_image': result_filename,
        'image_data': f"data:image/jpeg;base64,{img_data}"
    }
    
    # Log completion message
    if 'gemini_analysis' in summary:
        logger.info(f"Prediction complete with Gemini analysis. Found {summary['total']} masses.")
    else:
        logger.info(f"Prediction complete. Found {summary['total']} masses.")
    
    return jsonify(response), 200

@app.route('/api/results/<filename>', methods=['GET'])
def get_result(filename):
    """API endpoint to get result image"""
    return send_from_directory(RESULT_FOLDER, filename)

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    logger.info("Health check requested")
    return jsonify({'status': 'ok'}), 200

# Add a route for the root path
@app.route('/', methods=['GET'])
def index():
    """Root endpoint for testing"""
    logger.info("Root endpoint accessed")
    return jsonify({
        'status': 'ok',
        'message': 'Neuralrad Mammo AI is running',
        'endpoints': [
            '/api/health',
            '/api/predict',
            '/api/results/<filename>'
        ]
    }), 200

if __name__ == '__main__':
    # Initialize model on startup
    logger.info("Starting server and initializing model...")
    try:
        retinanet = load_model()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Warning: Failed to preload model: {e}")
        logger.info("Will attempt to load model on first request")
    
    # Log important paths
    logger.info(f"Upload folder: {UPLOAD_FOLDER}")
    logger.info(f"Results folder: {RESULT_FOLDER}")
    logger.info(f"Model path: {MODEL_PATH}")
    
    # Start Flask app
    logger.info(f"Starting Flask server on {SERVER_HOST}:{SERVER_PORT}")
    app.run(host=SERVER_HOST, port=SERVER_PORT, debug=DEBUG_MODE)
