import streamlit as st
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt

# Function to map images and apply noise
def map_image_with_noise(image, label):
    image = tf.cast(image, dtype=tf.float32) / 255.0
    noise_factor = 0.5
    noise = noise_factor * tf.random.normal(shape=image.shape)
    image_noisy = image + noise
    image_noisy = tf.clip_by_value(image_noisy, 0.0, 1.0)
    return image_noisy, image

# Function to define encoder
def encoder(inputs):
    conv_1 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same')(inputs)
    max_pool_1 = tf.keras.layers.MaxPooling2D((2,2))(conv_1)
    conv_2 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same')(max_pool_1)
    max_pool_2 = tf.keras.layers.MaxPooling2D((2,2))(conv_2)
    return max_pool_2

# Function to define bottleneck
def bottle_neck(inputs):
    bottleneck = tf.keras.layers.Conv2D(256, (3,3), activation='relu', padding='same')(inputs)
    encoder_visualization = tf.keras.layers.Conv2D(1, (3,3), activation='sigmoid', padding='same')(bottleneck)
    return bottleneck, encoder_visualization

# Function to define decoder
def decoder(inputs):
    conv_1 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same')(inputs)
    up_sample_1 = tf.keras.layers.UpSampling2D((2,2))(conv_1)
    conv_2 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same')(up_sample_1)
    up_sample_2 = tf.keras.layers.UpSampling2D((2,2))(conv_2)
    conv_3 = tf.keras.layers.Conv2D(1, (3,3), activation='sigmoid', padding='same')(up_sample_2)
    return conv_3

# Function to build the autoencoder model
def convolutional_auto_encoder():
    inputs = tf.keras.layers.Input(shape=(28, 28, 1,))
    encoder_output = encoder(inputs)
    bottleneck_output, encoder_visualization = bottle_neck(encoder_output)
    decoder_output = decoder(bottleneck_output)
    model = tf.keras.Model(inputs=inputs, outputs=decoder_output)
    encoder_model = tf.keras.Model(inputs=inputs, outputs=encoder_visualization)
    return model, encoder_model

# Streamlit interface
st.title('Convolutional Autoencoder for Fashion MNIST with Noisy Input')

# Load dataset without caching
def load_dataset():
    BATCH_SIZE = 128
    train_dataset = tfds.load('fashion_mnist', as_supervised=True, split="train")
    train_dataset = train_dataset.map(map_image_with_noise).batch(BATCH_SIZE).repeat()
    
    test_dataset = tfds.load('fashion_mnist', as_supervised=True, split="test")
    test_dataset = test_dataset.map(map_image_with_noise).batch(BATCH_SIZE).repeat()
    
    return train_dataset, test_dataset

train_dataset, test_dataset = load_dataset()

# Build and compile the model
convolutional_model, convolutional_encoder_model = convolutional_auto_encoder()
convolutional_model.compile(optimizer=tf.keras.optimizers.Adam(), loss='binary_crossentropy')

# Display model architecture
st.subheader("Model Architecture")
st.write(convolutional_model.summary())

# Train the model button
if st.button('Train the Model'):
    train_steps = 60000 // 128
    valid_steps = 10000 // 128
    conv_model_history = convolutional_model.fit(train_dataset, steps_per_epoch=train_steps, validation_data=test_dataset, validation_steps=valid_steps, epochs=5)
    st.success("Model trained successfully!")

# Function to display input, encoded, and decoded images
def display_results(input_images, encoded_images, decoded_images):
    fig, axes = plt.subplots(3, 10, figsize=(15, 5))
    for i in range(10):
        axes[0, i].imshow(input_images[i].reshape(28, 28), cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].imshow(encoded_images[i].reshape(7, 7), cmap='gray')
        axes[1, i].axis('off')
        axes[2, i].imshow(decoded_images[i].reshape(28, 28), cmap='gray')
        axes[2, i].axis('off')
    st.pyplot(fig)

# Test the model button
if st.button('Test the Model'):
    test_batch = next(iter(test_dataset.take(1)))
    input_images, _ = test_batch
    input_images = input_images.numpy()[:10]
    encoded_images = convolutional_encoder_model.predict(input_images)
    decoded_images = convolutional_model.predict(input_images)
    display_results(input_images, encoded_images, decoded_images)
