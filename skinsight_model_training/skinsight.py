#
# Import necessary libraries/dependancies
#

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import TopKCategoricalAccuracy





#
# Callback
#

# Define the checkpoint callback
checkpoint_callback_model = ModelCheckpoint(
    filepath='models/skinsight.keras',  # Path to save the model
    monitor='val_loss',        # Metric to monitor (e.g., 'val_loss', 'val_accuracy')
    save_best_only=True,       # Save only the model with the best monitored value
    save_weights_only=False,   # Save the entire model (architecture + weights)
    mode='min',                # Save model with the minimum 'val_loss'
    verbose=1                  # Print a message when saving the model
)

early_stopping = EarlyStopping(
    monitor='val_loss',   # Stop if validation loss stops improving
    patience=5,           # Wait 5 epochs before stopping
    restore_best_weights=True  # Restore best model weights
)






#
# Data normalization and loading to memory
#

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
validation_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    'data/train',  # Update with your path
    target_size=(224, 224),  # Resize images
    batch_size=32,
    class_mode='categorical',  # or 'sparse' depending on labels
    shuffle=True
)
validation_generator = validation_datagen.flow_from_directory(
    'data/val',  # Update with your path
    target_size=(224, 224),  # Resize images
    batch_size=32,
    class_mode='categorical',  # or 'sparse' depending on labels
    shuffle=False
)





#
# Loading the model
#

# Step 1: Load a Pre-trained Base Model
base_model = MobileNetV3Large(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False                                    # Freeze the base model (for transfer learning)

# Step 2: Add a Custom Classification Head
model = models.Sequential([
    base_model,                                                 # Pre-trained base    
    layers.GlobalAveragePooling2D(),                            # Pooling layer to reduce dimensions
    layers.Dropout(0.4),
    layers.Dense(10, activation='softmax')                      # 21 classes for skin diseases
])

# Define Top-3 Accuracy metric
top_3_accuracy = TopKCategoricalAccuracy(k=3)

# Step 3: Compile the Model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy', top_3_accuracy])

# Step 4: Print Model Summary
model.summary()





#
#
#

# Train the model
history = model.fit(
    train_generator,  # Training data
    epochs=30,  # Adjust epochs as needed
    validation_data=validation_generator,  # Validation data
    callbacks=[checkpoint_callback_model]
)


# Plot training and validation accuracy (visualization purposes)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title('Accuracy')
plt.show()

# Plot training and validation loss (visualization purposes)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Loss')
plt.show()





#
# Confusion Matrix
#

# Predict on the validation or test dataset
# Replace `valid_batches` with your validation/test data generator or dataset
y_true = validation_generator.classes  # True labels from the validation/test generator
y_pred_probs = model.predict(validation_generator)  # Predicted probabilities
y_pred = np.argmax(y_pred_probs, axis=1)  # Convert probabilities to class indices

# Calculate Precision, Recall, and F1-score
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-score: {f1}")

# Generate confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Define class labels
class_names = list(validation_generator.class_indices.keys())  # Assumes your generator has class indices

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# Print a detailed classification report
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))