import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.applications.efficientnet import preprocess_input  # Ensure this matches your model

# Load the TFLite model
model_path = "tflite_models/skinsight_10classes_topkacc.tflite"
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input & output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load and preprocess the image
image_path = "test/skibidi.jpg"
image = cv2.imread(image_path)

# Ensure the correct color format (BGR to RGB)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Resize to match the model's expected input shape (224x224 if using EfficientNet)
target_size = (input_details[0]['shape'][1], input_details[0]['shape'][2])
image = cv2.resize(image, target_size)

# Convert image to float32 (TFLite expects this)
image = image.astype(np.float32)

# Apply EfficientNet preprocessing (or replace with correct one if using another model)
image = preprocess_input(image)

# Add batch dimension
image = np.expand_dims(image, axis=0)

# Ensure correct dtype
image = image.astype(input_details[0]['dtype'])

# Run inference
interpreter.set_tensor(input_details[0]['index'], image)
interpreter.invoke()
predictions = interpreter.get_tensor(output_details[0]['index'])[0]

# Show class confidence scores
class_names = ["Acne",
"Actinic Keratoses",
"Benign Tumors",
"Candidiasis",
"Eczema",
"Leprosy",
"Rosacea",
"Skin Cancer",
"Vitiligo",
"Warts"]

print("\nClass Predictions:")
for i, score in enumerate(predictions):
    print(f"{class_names[i]}: {score*100:.2f}%")

# Show top predicted class
top_class = np.argmax(predictions)
print(f"\nTop Prediction: {class_names[top_class]} ({predictions[top_class]*100:.2f}%)")
