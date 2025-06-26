import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# Constants
IMAGE_SIZE = 256
CLASS_NAMES = ['Early Blight', 'Late Blight', 'Healthy']

# Load the model
model = tf.keras.models.load_model("../models/1.keras")

# Load the image
img_path = "v late.jpg"
img = image.load_img(img_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
img_array = image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create batch axis

# Predict
predictions = model.predict(img_array)
predicted_class = CLASS_NAMES[np.argmax(predictions)]

print(f"Predicted class: {predicted_class}")
