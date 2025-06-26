import streamlit as st
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

st.title("MNIST Autoencoder Demo")

# Normalize and reshape function
def preprocess_image(image):
    image = tf.cast(image, tf.float32) / 255.0
    return tf.reshape(image, (784,))

# Load MNIST dataset
BATCH_SIZE = 128
SHUFFLE_BUFFER_SIZE = 1024

train_dataset = tfds.load('mnist', split='train', as_supervised=True)
train_dataset = train_dataset.map(lambda x, y: preprocess_image(x)).shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE).repeat()

# Define the autoencoder
def build_autoencoder():
    input_layer = tf.keras.Input(shape=(784,))
    encoded = tf.keras.layers.Dense(128, activation='relu')(input_layer)
    encoded = tf.keras.layers.Dense(64, activation='relu')(encoded)
    encoded = tf.keras.layers.Dense(32, activation='relu')(encoded)

    decoded = tf.keras.layers.Dense(64, activation='relu')(encoded)
    decoded = tf.keras.layers.Dense(128, activation='relu')(decoded)
    decoded = tf.keras.layers.Dense(784, activation='sigmoid')(decoded)

    encoder = tf.keras.Model(inputs=input_layer, outputs=encoded)
    autoencoder = tf.keras.Model(inputs=input_layer, outputs=decoded)
    return encoder, autoencoder

# Train model (for demo, you might skip this step and use pre-trained weights)
encoder, autoencoder = build_autoencoder()
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

train_steps = 60000 // BATCH_SIZE
autoencoder.fit(train_dataset, steps_per_epoch=train_steps, epochs=1)

# Upload image
uploaded_file = st.file_uploader("Upload a 28x28 grayscale image (png/jpg/jpeg):", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L').resize((28, 28))
    image_array = np.array(image) / 255.0
    image_array = image_array.reshape((1, 784))

    encoded_output = encoder.predict(image_array)
    decoded_output = autoencoder.predict(image_array)

    st.subheader("Original Image")
    st.image(image, width=150)

    st.subheader("Encoded Vector")
    st.write(encoded_output)

    st.subheader("Reconstructed Image")
    st.image(decoded_output.reshape(28, 28), width=150)
