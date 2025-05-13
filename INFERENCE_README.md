# Object Detection with PyTorch RetinaNet

This guide explains how to run inference on a single image using the PyTorch RetinaNet model.

## Quick Start

1. **Setup your Python environment** with PyTorch, OpenCV, and other dependencies
2. **Run the inference script**:
   ```bash
   python inference.py --image test.jpeg --output result.jpg
   ```

## Environment Setup

Choose one of these options to set up your environment:

### Option 1: Using Conda
```bash
conda create -n retinanet python=3.8
conda activate retinanet
pip install torch torchvision numpy opencv-python
```

### Option 2: Using venv
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install torch torchvision numpy opencv-python
```

## Running Inference

The `inference.py` script performs object detection on a single image using the RetinaNet model with COCO pretrained weights.

### Command-line Arguments

```bash
python inference.py --image <path_to_image> --output <path_to_save_output> --score-threshold <confidence_threshold>
```

- `--image`: Path to the input image (required)
- `--model`: Path to the model checkpoint (default: `coco_resnet_50_map_0_335_state_dict.pt`)
- `--score-threshold`: Confidence threshold for detections (default: 0.5)
- `--output`: Path to save the output image (default: `output.jpg`)

### Example

```bash
python inference.py --image test.jpeg --output result.jpg --score-threshold 0.4
```

## How it Works

1. The script loads the RetinaNet model with pretrained weights
2. It preprocesses the input image (resizing and normalizing)
3. Runs inference through the model to detect objects
4. Draws bounding boxes and labels on the image
5. Saves the annotated image to the output path

## COCO Classes

The model is pretrained on COCO dataset with 80 classes, including common objects like person, car, dog, etc.
