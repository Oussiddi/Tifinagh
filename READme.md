# Tifinagh Character Recognition System

## Overview
An advanced machine learning system designed to recognize and digitize handwritten Tifinagh characters and words. The system supports both individual character recognition and full word processing, making it a valuable tool for preserving and processing Tifinagh script.

## Features

### 1. Dual Recognition Modes
- **Single Character Recognition**
  - Upload individual character images
  - Get instant character recognition with confidence scores
  - Supports image rotation for better accuracy
  - Shows both Latin and Tifinagh script outputs

- **Word Recognition**
  - Processes complete word images
  - Automatic character segmentation
  - Shows step-by-step recognition process

### 2. User Interface
- Clean, intuitive web interface built with Streamlit
- Real-time processing and results display
- Image preview and manipulation options
- Clear confidence scores and predictions

### 3. Character Support
Supports 33 Tifinagh characters including:
```
ⴰ ⴱ ⵛ ⴷ ⴹ ⵄ ⴼ ⴳ ⵖ ⴳⵯ ⵀ ⵃ ⵊ ⴽ ⴽⵯ ⵍ ⵎ ⵏ ⵇ ⵔ 
ⵕ ⵙ ⵚ ⵜ ⵟ ⵡ ⵅ ⵢ ⵣ ⵥ ⴻ ⵉ ⵓ
```
### 4 . Project Structure
```
tifinagh-recognition/
├── a2.py             # Character recognition model
├── front2.py         # Streamlit interface
├── model_2.h5        # Trained model weights
├── model_3.h5        # Additional model weights
└── tifinagh_v2.ipynb # Development notebook
```
## Setup and Installation

### Prerequisites
```bash
python 3.7+
pip
```

### Installation Steps
1. Clone the repository:
```bash
git clone https://github.com/yourusername/tifinagh-recognition.git
cd tifinagh-recognition
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Download the model:
```bash
# Option 1: Download from releases page
# Option 2: Use git-lfs
git lfs install
git lfs pull
```

### Running the Application
1. Start the web interface:
```bash
streamlit run front.py
```

2. Access the application:
- Open your browser
- Navigate to `http://localhost:8501`

## Usage Guide

### Single Character Recognition
1. Select "Character" mode in the sidebar
2. Upload an image of a single Tifinagh character
3. Use the rotate button if needed
4. View the prediction and confidence score

### Word Recognition
1. Select "Word" mode in the sidebar
2. Upload an image containing Tifinagh text
3. View the segmentation results
4. See the complete word recognition

### Best Practices for Image Upload
- Clear, well-lit images
- Dark text on light background
- Supported formats: JPG, JPEG, PNG

## Performance

### Recognition Accuracy
- ~98% accuracy


## Dataset and Training

The system was trained on a comprehensive dataset of Tifinagh characters, including:
- Handwritten samples
- Various writing styles
- Different pen types and thicknesses
- Multiple writers


## Support

- Contact me 
