import streamlit as st
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Function to normalize and flatten the image
def map_image(image):
    image = tf.cast(image, dtype=tf.float32)
    image = image / 255.0
    image = tf.reshape(image, shape=(784,))
    return image

# Load the dataset
BATCH_SIZE = 128
SHUFFLE_BUFFER_SIZE = 1024

train_dataset = tfds.load('mnist', as_supervised=True, split="train")
train_dataset = train_dataset.map(lambda x, y: map_image(x)).shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE).repeat()

# Define the autoencoder model
def deep_autoencoder():
    inputs = tf.keras.layers.Input(shape=(784,))
    encoder = tf.keras.layers.Dense(units=128, activation='relu')(inputs)
    encoder = tf.keras.layers.Dense(units=64, activation='relu')(encoder)
    encoder = tf.keras.layers.Dense(units=32, activation='relu')(encoder)

    decoder = tf.keras.layers.Dense(units=64, activation='relu')(encoder)
    decoder = tf.keras.layers.Dense(units=128, activation='relu')(decoder)
    decoder = tf.keras.layers.Dense(units=784, activation='sigmoid')(decoder)

    return tf.keras.Model(inputs=inputs, outputs=encoder), tf.keras.Model(inputs=inputs, outputs=decoder)

# Instantiate models
deep_encoder_model, deep_autoencoder_model = deep_autoencoder()

# Compile the model
deep_autoencoder_model.compile(optimizer=tf.keras.optimizers.Adam(), loss='binary_crossentropy')

# Train the model
train_steps = 60000 // BATCH_SIZE
deep_autoencoder_model.fit(train_dataset, steps_per_epoch=train_steps, epochs=50)

# Streamlit UI
st.title("MNIST Autoencoder")
st.write("Upload an image to see its encoded and decoded versions.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Process the uploaded image
    image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28
    image_array = np.array(image)
    image_array = image_array / 255.0  # Normalize
    image_array = np.reshape(image_array, (1, 784))  # Flatten

    # Make predictions
    encoded = deep_encoder_model.predict(image_array)
    decoded = deep_autoencoder_model.predict(image_array)

    # Display the results
    st.subheader("Original Image")
    st.image(image, caption='Uploaded Image', use_column_width=True)

    st.subheader("Encoded Output")
    st.write(encoded)

    decoded_image = np.reshape(decoded, (28, 28))
    st.subheader("Decoded Image")
    st.image(decoded_image, caption='Decoded Image', use_column_width=True)

