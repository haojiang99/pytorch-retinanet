# PyTorch RetinaNet Project

## Project Overview

This project is a PyTorch implementation of the RetinaNet object detection model as described in the paper [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002). The implementation has been customized for medical imaging, specifically for mammogram analysis to detect breast cancer.

### Key Components

The project consists of three main parts:

1. **Core RetinaNet Model**: Located in the `retinanet/` directory, this contains the PyTorch implementation of the RetinaNet architecture.
2. **Server Application**: Located in the `server/` directory, this is a Flask API that serves the model for inference.
3. **Client Application**: Located in the `client/` directory, this is a Svelte-based web interface for uploading mammograms and viewing results.

## Architecture

- **Model**: Uses ResNet (18, 34, 50, 101, or 152) as a backbone with Feature Pyramid Network (FPN)
- **Server**: Flask API for serving predictions
- **Client**: Svelte web application with file upload and result visualization

## Project Structure

```
pytorch-retinanet/
├── retinanet/                # Core model implementation
│   ├── model.py              # Main RetinaNet model architecture
│   ├── anchors.py            # Anchor box generation
│   ├── losses.py             # Focal loss implementation
│   └── utils.py              # Utility functions
├── server/                   # Flask server for model serving
│   ├── app.py                # Main server application
│   ├── config.py             # Server configuration
│   ├── gemini_client.py      # Optional Gemini AI interpretation
│   └── requirements.txt      # Server dependencies
├── client/                   # Svelte web client
│   ├── src/                  # Client source code
│   │   ├── App.svelte        # Main application component
│   │   └── routes/           # Application routes
│   │       ├── Home.svelte   # Image upload page
│   │       └── Results.svelte # Results display page
│   └── public/               # Static assets
├── train.py                  # Training script for RetinaNet
├── inference.py              # General inference script
└── inference_ddsm.py         # Mammogram-specific inference
```

## Model Details

The RetinaNet model architecture includes:

- ResNet backbone (configurable depth: 18, 34, 50, 101, 152)
- Feature Pyramid Network (FPN) for multi-scale feature extraction
- Separate classification and regression subnetworks
- Focal loss for handling class imbalance
- Anchor boxes for detecting objects at multiple scales

## Usage Commands

### Training

For training on the COCO dataset:
```
python train.py --dataset coco --coco_path /path/to/coco --depth 50
```

For training using a custom CSV dataset:
```
python train.py --dataset csv --csv_train /path/to/train_annots.csv --csv_classes /path/to/train/class_list.csv --csv_val /path/to/val_annots.csv
```

### Validation

For COCO dataset validation:
```
python coco_validation.py --coco_path /path/to/coco --model_path /path/to/model.pt
```

For CSV dataset validation:
```
python csv_validation.py --csv_annotations_path /path/to/annotations.csv --model_path /path/to/model.pt --images_path /path/to/images_dir --class_list_path /path/to/class_list.csv
```

### Inference

For general object detection:
```
python inference.py --image /path/to/image.jpg --model /path/to/model.pt --score-threshold 0.5
```

For mammogram analysis:
```
python inference_ddsm.py --image /path/to/mammogram.jpg --model /path/to/ddsm_model.pt --score-threshold 0.3
```

### Visualization

```
python visualize.py --dataset csv --csv_classes /path/to/class_list.csv --csv_val /path/to/val_annots.csv --model /path/to/model.pt
```

### Running the Server

```
cd server
python app.py
```

### Running the Client

```
cd client
npm install  # Only needed first time
npm run dev
```

## Data Format

The project supports two dataset formats:

1. **COCO**: Standard MS COCO format
2. **CSV**: Custom format with annotations in CSV:
   ```
   path/to/image.jpg,x1,y1,x2,y2,class_name
   ```

   Class mapping in separate CSV:
   ```
   class_name,id
   ```

## Medical Application

The project has been adapted for breast cancer detection in mammograms with four classes:
- Benign mass
- Malignant mass
- Benign calcification
- Malignant calcification

## Dependencies

- Python 3.7+
- PyTorch
- OpenCV
- NumPy
- Flask (for server)
- Node.js and npm (for client)
- Svelte (for client)

## Model Weights

Pre-trained models:
- General purpose: Available in the README
- Mammogram-specific: Stored in `ddsm_checkpoints/`

## Acknowledgements

- Original RetinaNet paper: [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
- Based on [keras retinanet implementation](https://github.com/fizyr/keras-retinanet)
- Uses NMS module from [pytorch faster-rcnn implementation](https://github.com/ruotianluo/pytorch-faster-rcnn)