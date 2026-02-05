"""
CNN Dog vs Cat Classifier
-------------------------
This script builds, trains, and tests a Convolutional Neural Network
using TensorFlow/Keras.

Make sure:
- Dataset paths exist
- TensorFlow is installed
- Images are organized in class folders
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image


# =========================
# 1. Build CNN Model
# =========================
cnn = Sequential()

# First Convolution Layer + Pooling
cnn.add(Conv2D(
    filters=32,
    kernel_size=(3, 3),
    input_shape=(64, 64, 3),
    activation="relu"
))
cnn.add(MaxPooling2D(pool_size=(2, 2)))

# Second Convolution Layer + Pooling
cnn.add(Conv2D(16, (3, 3), activation="relu"))
cnn.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten layer converts 2D feature maps into 1D vector
cnn.add(Flatten())

# Fully Connected ANN layers
cnn.add(Dense(64, activation="relu"))
cnn.add(Dense(32, activation="relu"))
cnn.add(Dense(16, activation="relu"))
cnn.add(Dense(8, activation="relu"))
cnn.add(Dense(4, activation="relu"))

# Output layer (binary classification: dog vs cat)
cnn.add(Dense(1, activation="sigmoid"))

# Compile model
cnn.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

print("Model compiled successfully.")


# =========================
# 2. Image Data Generators
# =========================
# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Only rescaling for test data
test_datagen = ImageDataGenerator(rescale=1.0 / 255)


# =========================
# 3. Load Dataset
# =========================
train_generator = train_datagen.flow_from_directory(
    r'D:\Deep_Learning_WS_Cube_Tech\Convolutional_Neural_Network_CNN\dog vs cat\dataset\training_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    r'D:\Deep_Learning_WS_Cube_Tech\Convolutional_Neural_Network_CNN\dog vs cat\dataset\test_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)


# =========================
# 4. Train the Model
# =========================
print("Training started...")
cnn.fit(
    train_generator,
    steps_per_epoch=200,
    epochs=5,
    validation_data=test_generator
)

print("Training completed.")


# =========================
# 5. Predict Single Image
# =========================
img_path = r'D:\Deep_Learning_WS_Cube_Tech\Convolutional_Neural_Network_CNN\dog vs cat\dataset\test_set\cats\cat.4001.jpg'

# Load and preprocess image
img = image.load_img(img_path, target_size=(64, 64))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)

# Prediction
prediction = cnn.predict(img)

print("Raw prediction value:", prediction)

# Interpret result
if prediction[0][0] < 0.5:
    print("Prediction: Dog")
else:
    print("Prediction: Cat")
