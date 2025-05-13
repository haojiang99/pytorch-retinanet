import os
import base64
import cv2
import numpy as np
import requests
import json
import logging
from PIL import Image
from io import BytesIO

# Import config
from config import GEMINI_API_KEY

# Configure logging
logger = logging.getLogger(__name__)

class GeminiClient:
    """Client for Google Gemini 2.0 Flash API integration"""
    
    def __init__(self, api_key=None):
        """Initialize Gemini client with API key"""
        self.api_key = api_key or GEMINI_API_KEY or os.environ.get('GEMINI_API_KEY')
        if not self.api_key:
            logger.warning("No Gemini API key provided. Set GEMINI_API_KEY environment variable.")
        
        self.api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    
    def resize_image(self, image, max_side=1200):
        """Resize image to have max side length of max_side"""
        # Convert to PIL Image if it's a numpy array
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        elif isinstance(image, str):
            # If it's a file path
            image = Image.open(image)
        
        # Get current dimensions
        width, height = image.size
        
        # Determine scale factor
        if width > height:
            scale = max_side / width
        else:
            scale = max_side / height
        
        # Only scale down, not up
        if scale < 1:
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
        return image
    
    def encode_image(self, image):
        """Convert image to base64 string for API request"""
        if isinstance(image, Image.Image):
            buffer = BytesIO()
            image.save(buffer, format="JPEG")
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
        elif isinstance(image, str) and os.path.isfile(image):
            with open(image, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode('utf-8')
        elif isinstance(image, np.ndarray):
            # Convert OpenCV image to base64
            success, buffer = cv2.imencode(".jpg", image)
            if not success:
                raise ValueError("Failed to encode image")
            return base64.b64encode(buffer).decode('utf-8')
        else:
            raise ValueError("Unsupported image format")
    
    def construct_prompt(self, prediction_results):
        """Construct prompt for Gemini based on RetinaNet prediction results"""
        prompt = """
You are an AI radiologist assistant specialized in mammogram analysis. 
Please analyze this mammogram image and provide a detailed reading. 

RetinaNet prediction results:
"""
        
        if not prediction_results:
            prompt += "No masses detected by RetinaNet.\n"
        else:
            prompt += f"Total findings: {prediction_results.get('total', 0)}\n"
            prompt += f"Benign masses: {prediction_results.get('benign', 0)}\n"
            prompt += f"Malignant masses: {prediction_results.get('malignant', 0)}\n\n"
            
            if 'findings' in prediction_results and prediction_results['findings']:
                prompt += "Detected masses:\n"
                for i, finding in enumerate(prediction_results['findings']):
                    box = finding.get('box', [0, 0, 0, 0])
                    width = box[2] - box[0]
                    height = box[3] - box[1]
                    area = width * height
                    
                    prompt += f"Finding {i+1}: {finding.get('class', 'unknown')} "
                    prompt += f"(confidence: {finding.get('score', 0):.3f}, "
                    prompt += f"size: {width}x{height} pixels, area: {area} pixels)\n"
        
        prompt += """
Please provide:
1. Identification of any suspicious masses or calcifications (describe their appearance, location, and characteristics)
2. Assessment of the findings (e.g., BI-RADS category if possible)
3. Any recommendations for follow-up or additional imaging that would be appropriate
4. Overall impression

Focus on being accurate, clear, and concise in your assessment.
"""
        return prompt
    
    def analyze_mammogram(self, image, prediction_results):
        """Send mammogram image and prediction results to Gemini for analysis"""
        if not self.api_key:
            logger.error("Gemini API key not configured")
            return {"error": "Gemini API key not configured"}
        
        try:
            # Resize image to max side < 1200
            resized_image = self.resize_image(image)
            
            # Convert image to base64
            image_base64 = self.encode_image(resized_image)
            
            # Construct prompt for Gemini
            prompt = self.construct_prompt(prediction_results)
            
            # Prepare request data
            request_data = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {"text": prompt},
                            {
                                "inline_data": {
                                    "mime_type": "image/jpeg",
                                    "data": image_base64
                                }
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.2,
                    "maxOutputTokens": 1024,
                    "topP": 0.8
                }
            }
            
            # Make API request to Gemini
            response = requests.post(
                f"{self.api_url}?key={self.api_key}",
                json=request_data,
                headers={"Content-Type": "application/json"}
            )
            
            # Check if request was successful
            if response.status_code == 200:
                result = response.json()
                
                # Extract the text from the response
                if "candidates" in result and len(result["candidates"]) > 0:
                    if "content" in result["candidates"][0]:
                        content = result["candidates"][0]["content"]
                        if "parts" in content and len(content["parts"]) > 0:
                            return {
                                "success": True,
                                "gemini_analysis": content["parts"][0]["text"]
                            }
            
            # If we reached here, something went wrong
            logger.error(f"Gemini API error: {response.status_code}")
            logger.error(f"Response: {response.text}")
            return {
                "success": False,
                "error": f"Gemini API error: {response.status_code}",
                "details": response.text
            }
                
        except Exception as e:
            logger.error(f"Error in Gemini analysis: {str(e)}")
            return {
                "success": False,
                "error": f"Error in Gemini analysis: {str(e)}"
            }
