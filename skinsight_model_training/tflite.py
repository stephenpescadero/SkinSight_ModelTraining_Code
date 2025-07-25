import os
import tensorflow as tf

# Paths
keras_models_dir = "models"
tflite_models_dir = "tflite_models"

# Ensure output directory exists
os.makedirs(tflite_models_dir, exist_ok=True)

# Convert each .keras model to .tflite
for model_file in os.listdir(keras_models_dir):
    if model_file.endswith(".keras"):
        model_path = os.path.join(keras_models_dir, model_file)
        tflite_model_path = os.path.join(tflite_models_dir, model_file.replace(".keras", ".tflite"))

        # Load the Keras model
        model = tf.keras.models.load_model(model_path)

        # Convert model to TFLite format
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()

        # Save the TFLite model
        with open(tflite_model_path, "wb") as f:
            f.write(tflite_model)

        print(f"Converted {model_file} -> {tflite_model_path}")

print("All models converted successfully!")
