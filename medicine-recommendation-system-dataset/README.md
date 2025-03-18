# Medical Symptom-Based Disease Prediction System

A comprehensive system for predicting potential medical conditions based on symptoms, providing severity-based analysis and personalized health recommendations using machine learning.

## Features

- 🏥 Intelligent symptom-based disease prediction
- 📊 Three-level severity classification (Mild, Moderate, Severe)
- 💬 Natural language symptom input support
- 📋 Comprehensive health recommendations
- 🥗 Personalized dietary and lifestyle advice

## Project Structure
```
medical-diagnosis-system/
│
├── dataset/
│   ├── Training-weighted.csv     # Main training dataset
│   ├── description.csv          # Disease descriptions
│   ├── precautions_df.csv      # Medical precautions
│   ├── medications.csv         # Medication guidelines
│   ├── diets.csv              # Dietary recommendations
│   └── workout_df.csv         # Exercise guidelines
│
├── interactive_diagnosis.py    # Main application
└── README.md
```

## Dataset & Model Performance

### Dataset Overview
- Training Dataset: 4920+ records with 132 symptoms and 41 diseases
- Supporting Data: Disease descriptions, precautions, medications, diet, and exercise recommendations

### Accuracy Metrics
- Overall Accuracy: 95%
- Precision: 93%
- Recall: 92%
- F1-Score: 92.5%

## Tech Stack

- Python 3.x
- Pandas, NumPy, Scikit-learn
- RandomForest Classifier

## Installation & Usage

1. Install required packages:
```bash
pip install pandas numpy scikit-learn
```

2. Run the system:
```bash
python interactive_diagnosis.py
```

3. Enter symptoms when prompted:
```
Enter your symptoms (comma-separated), or 'quit' to exit:
fever, headache, fatigue
```

Example Output:
```
POSSIBLE CONDITIONS
==================
MILD LEVEL:
- Common Cold (Very Likely)
- Sinusitis (Possible)

MODERATE LEVEL:
- Viral Fever (Less Likely)

GENERAL RECOMMENDATIONS
======================
[Personalized health advice and recommendations]
```

## Important Notes

- For educational and informational purposes only
- Not a substitute for professional medical advice
- Always consult healthcare providers for proper diagnosis
- In case of emergency, contact medical services immediately 