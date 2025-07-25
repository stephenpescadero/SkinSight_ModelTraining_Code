import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained model
model_path = "models/skinsight.keras"  # Change this to your model path
model = tf.keras.models.load_model(model_path)

# Path to test dataset (organized into labeled subdirectories)
test_dir = "data/test"  # Change this to your test dataset path

# Load test dataset
batch_size = 32  # Adjust based on memory
img_size = (model.input_shape[1], model.input_shape[2])  # Get model's input size

test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    image_size=img_size,
    batch_size=batch_size,
    shuffle=False,  # Don't shuffle so labels remain correct
    label_mode="categorical"  # Categorical for one-hot encoding
)

# Get class names from the dataset
class_names = test_dataset.class_names
print("Classes found:", class_names)

# Evaluate the model on test dataset
results = model.evaluate(test_dataset)
print(f"\nTest Loss: {results[0]:.4f}")
print(f"Test Accuracy: {results[1] * 100:.2f}%")
print(f"Top-3 Accuracy: {results[2] * 100:.2f}%")  # Assuming Top-3 Accuracy was added in model.compile

# Get predictions for all test images
predictions = model.predict(test_dataset)

# Calculate Top-3 Accuracy manually
true_labels = np.argmax(np.concatenate([y for x, y in test_dataset], axis=0), axis=1)
top_k_predictions = np.argsort(predictions, axis=1)[:, -3:]  # Top-3 indices for each prediction

top_3_correct = sum(true_labels[i] in top_k_predictions[i] for i in range(len(true_labels)))
top_3_accuracy = top_3_correct / len(true_labels)

print(f"\nManual Top-3 Accuracy: {top_3_accuracy * 100:.2f}%")

# Calculate other metrics (e.g., Precision, Recall, F1-score)
predicted_labels = np.argmax(predictions, axis=1)  # Top-1 predictions
print("\nClassification Report:")
print(classification_report(true_labels, predicted_labels, target_names=class_names))

# Generate a confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
