# Brain Tumor MRI Detection Web Application

A modern web application for detecting brain tumors in MRI scans using deep learning. The application provides real-time analysis of MRI images with confidence scores for different types of brain tumors.

## Features

- 🧠 Accurate brain tumor detection from MRI scans
- 📊 Confidence scores for multiple tumor types
- 🖼️ Support for common image formats (PNG, JPG, JPEG)
- 🎯 Real-time analysis with visual feedback
- 📱 Responsive design for all devices
- 🔄 Drag and drop file upload
- ⚡ Fast and efficient processing

## Dataset

This project uses the Brain Tumor MRI Dataset from Kaggle:
[Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

The dataset contains 7023 MRI images of human brains, classified into 4 classes:
- Glioma
- Meningioma
- Pituitary
- No Tumor

### Dataset Details
- Total Images: 7023
- Image Format: jpg
- Resolution: 240 x 240 pixels
- Color Mode: RGB and Grayscale

## Tech Stack

- **Frontend**: HTML5, CSS3, JavaScript
- **Backend**: Python, Flask
- **Deep Learning**: TensorFlow, Keras
- **Image Processing**: OpenCV, Pillow
- **UI Components**: Font Awesome, Inter Font

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/brain-tumor-detection.git
cd brain-tumor-detection
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the model:
- Visit [CNN Brain Tumor Detector on Kaggle](https://www.kaggle.com/models/esfiam/cnn-brain-tumor-detector?select=brain_tumor_mri.keras)
- Download the `brain_tumor_mri.keras` file
- Place it in the project root directory

5. Run the application:
```bash
python app.py
```

6. Open your browser and navigate to:
```
http://localhost:5000
```

## Project Structure

```
brain-tumor-detection/
│
├── static/
│   ├── style.css
│   └── uploads/
│
├── templates/
│   ├── index.html
│   └── result.html
│
├── app.py
├── requirements.txt
└── README.md
```

## Usage

1. Access the web interface through your browser
2. Upload an MRI scan image (drag and drop or click to browse)
3. Click "Analyze Image"
4. View the results showing:
   - Predicted tumor type
   - Confidence scores for each class
   - Original uploaded image

## Model Performance

The model achieves the following performance metrics:
- Accuracy: ~98%
- Precision: ~97%
- Recall: ~97%
- F1-Score: ~97%

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

