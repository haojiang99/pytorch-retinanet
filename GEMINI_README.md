# Mammogram Analysis with Gemini AI

This extension adds [Google Gemini 2.0 Flash](https://ai.google.dev/gemini) integration to the mammogram analysis system, providing advanced interpretations of mammogram images.

## Features

- Mammogram analysis with RetinaNet deep learning model
- Enhanced interpretation with Google Gemini 2.0 Flash
- Resizing of images to max side < 1200 for Gemini compatibility
- Detailed radiologist-like analysis of suspicious masses or calcifications

## Setup

1. Obtain a [Google Gemini API key](https://ai.google.dev/tutorials/setup)
2. Add your API key to one of these options:
   - Edit `server/gemini_api_key.txt` with your key
   - Set the `GEMINI_API_KEY` environment variable
   - Create a `.env` file in the server directory with `GEMINI_API_KEY=your_key_here`

## Configuration

Edit `server/config.py` to customize:
- Server host and port
- Debug mode
- Default model confidence threshold
- Whether to use Gemini by default

## Using the Application

1. Start the server with `python server/app.py`
2. Open the client application at `http://localhost:5001`
3. Upload a mammogram image
4. Toggle the "Use Gemini AI" option as needed
5. View detailed analysis results, including:
   - RetinaNet mass detection
   - Mass classification (benign vs. malignant)
   - Gemini AI radiologist-like interpretation
   - BI-RADS assessment and recommendations

## Technical Details

### Image Processing Flow

1. The system processes the mammogram with RetinaNet to detect and classify masses
2. The image is resized to have a maximum dimension of 1200 pixels
3. The resized image and RetinaNet predictions are sent to Gemini 2.0 Flash
4. Gemini provides radiologist-like interpretation of the findings
5. The combined results are displayed to the user

### API Endpoints

- `POST /api/predict`: Upload and analyze a mammogram image
  - Parameters:
    - `file`: Mammogram image file (JPEG/PNG)
    - `threshold`: Detection confidence threshold (optional, default=0.3)
    - `use_gemini`: Whether to use Gemini analysis (optional, default=true)

## Dependencies

- Python 3.8+
- PyTorch
- Flask
- Google Gemini API (requires an API key)
- Pillow
- OpenCV
- NumPy
- Svelte (client-side)

## Troubleshooting

- If Gemini analysis is not working, check:
  - Your API key is correctly set
  - Network connectivity to the Google API
  - The server logs for detailed error messages
