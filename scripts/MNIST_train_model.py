import tensorflow as tf
import numpy as np

# Load and preprocess MNIST data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize to [0,1]

# Build a simple 3-layer neural network
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),     # 784 inputs
    tf.keras.layers.Dense(128, activation='relu'),      # Hidden layer
    tf.keras.layers.Dense(64, activation='relu'),       # Hidden layer  
    tf.keras.layers.Dense(10, activation='softmax')     # 10 digit classes
])

# Compile and train
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, batch_size=32)

# Export for C inference (we'll need this later)
model.save_weights('mnist.weights.h5')
print(f"Test accuracy: {model.evaluate(x_test, y_test)[1]:.3f}")