# Mammography Analysis with RetinaNet

This application provides a web-based interface for detecting and classifying masses in mammogram images using the RetinaNet deep learning model.

## System Overview

The application consists of two main components:

1. **Flask Server**: A Python backend that serves the RetinaNet model for inference
2. **Svelte.js Client**: A web frontend for uploading images and viewing results

## Project Structure

```
pytorch-retinanet/
├── client/                 # Svelte.js client application
├── server/                 # Flask server application
├── retinanet/              # RetinaNet model code
├── model_final.pt          # Trained model weights
└── ...
```

## Setup and Installation

### Server Setup

1. Navigate to the server directory:

```bash
cd server
```

2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Run the server:

```bash
python app.py
```

The server will be available at http://localhost:5000

### Client Setup

1. Navigate to the client directory:

```bash
cd client
```

2. Install the dependencies:

```bash
npm install
```

3. Start the development server:

```bash
npm run dev
```

The client will be available at http://localhost:3000

## Usage

1. Open the client application in your browser at http://localhost:3000
2. Upload a mammogram image (JPEG or PNG format)
3. The image will be processed by the model, and results will be displayed with annotations
4. View the analysis summary, which includes:
   - Number of detected masses
   - Classification (benign or malignant)
   - Confidence scores

## Model Information

This application uses a RetinaNet model with the following characteristics:

- Based on ResNet-50 backbone
- Trained on DDSM (Digital Database for Screening Mammography) dataset
- Detects and classifies mammogram masses as either benign or malignant

## Development

- **Server**: The Flask API is defined in `server/app.py`
- **Client**: The Svelte.js application is in the `client/` directory
- **Model**: The RetinaNet implementation is in the `retinanet/` directory

## License

This project is licensed under the MIT License - see the LICENSE file for details.
