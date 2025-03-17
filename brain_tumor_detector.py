import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the model
model = load_model('brain_tumor_mri.keras')

# Your image path
image_path = 'D:/06_projects/Impetus/image.png'

# Load and preprocess the image
img = load_img(image_path, target_size=(256, 256), color_mode='grayscale')  
img_array = img_to_array(img)

# Simple normalization
img_normalized = img_array / 255.0
img_normalized = np.expand_dims(img_normalized, axis=0)

# Make predictions
predictions = model.predict(img_normalized)

# Define classes
classes = ['Glioma', 'Meningioma', 'Pituitary', 'No Tumor']

# Print results
print("\nPrediction Results:")
print("Raw predictions:", predictions[0])
predicted_class = np.argmax(predictions, axis=1)
print(f"\nPredicted Class: {classes[predicted_class[0]]}")
print("\nConfidence Scores:")
for class_name, prob in zip(classes, predictions[0]):
    print(f"{class_name}: {prob*100:.2f}%")

# Print model summary to verify input shape
print("\nModel Architecture:")
model.summary()
