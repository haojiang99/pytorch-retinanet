# RetinaNet Mammography Server

This Flask server provides an API endpoint for mammogram image detection using a PyTorch RetinaNet model.

## Setup

1. Install the required packages:

```bash
pip install -r requirements.txt
```

2. Run the server:

```bash
python app.py
```

The server will be available at http://localhost:5000

## API Endpoints

### Health Check
- **URL**: `/api/health`
- **Method**: `GET`
- **Response**: JSON with status message

### Predict
- **URL**: `/api/predict`
- **Method**: `POST`
- **Parameters**:
  - `file`: The mammogram image file (jpg, jpeg, or png)
  - `threshold` (optional): Confidence threshold for detections (default: 0.3)
- **Response**: JSON with detection results and base64-encoded annotated image

### Get Result Image
- **URL**: `/api/results/<filename>`
- **Method**: `GET`
- **Response**: The processed image file with annotations

## Response Format

The prediction response includes:

```json
{
  "summary": {
    "total": 2,
    "benign": 1,
    "malignant": 1,
    "findings": [
      {
        "class": "benign",
        "score": 0.95,
        "box": [100, 100, 200, 200],
        "area": 10000
      },
      ...
    ],
    "highest_confidence": {
      "class": "benign",
      "score": 0.95,
      "box": [100, 100, 200, 200],
      "area": 10000
    },
    "largest_mass": {
      "class": "malignant",
      "score": 0.87,
      "box": [300, 300, 500, 500],
      "area": 40000
    }
  },
  "result_image": "result_uuid.jpg",
  "image_data": "data:image/jpeg;base64,..."
}
```
