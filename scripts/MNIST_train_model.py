import tensorflow as tf
import numpy as np
import os

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

# Export weights in binary format for C inference
def export_weights_for_c(model, filename):
   """Export model weights as binary file for C inference"""
   weights = model.get_weights()
   
   # Expected order: weights1, bias1, weights2, bias2, weights3, bias3
   with open(filename, 'wb') as f:
       # Layer 1: Dense weights (784x128) and bias (128,)
       f.write(weights[0].astype(np.float32).tobytes())  # weights1
       f.write(weights[1].astype(np.float32).tobytes())  # bias1
       
       # Layer 2: Dense weights (128x64) and bias (64,)  
       f.write(weights[2].astype(np.float32).tobytes())  # weights2
       f.write(weights[3].astype(np.float32).tobytes())  # bias2
       
       # Layer 3: Dense weights (64x10) and bias (10,)
       f.write(weights[4].astype(np.float32).tobytes())  # weights3
       f.write(weights[5].astype(np.float32).tobytes())  # bias3
   
   print(f"Exported weights to {filename}")
   
   # Print weight shapes for verification
   for i, w in enumerate(weights):
       print(f"Layer {i//2 + 1} {'weights' if i%2==0 else 'bias'}: {w.shape}")

# Create test data files for C inference using TensorFlow's data
def create_test_files_for_c():
   """Create MNIST test files in the format C code expects"""
   
   # Create images file
   with open('t10k-images-idx3-ubyte', 'wb') as f:
       # MNIST image file header: magic number, num images, rows, cols
       f.write(np.array([0x00000803, 100, 28, 28], dtype='>i4').tobytes())
       
       # Write first 100 test images as uint8
       for i in range(100):
           img = (x_test[i] * 255).astype(np.uint8)
           f.write(img.tobytes())
   
   # Create labels file  
   with open('t10k-labels-idx1-ubyte', 'wb') as f:
       # MNIST label file header: magic number, num labels
       f.write(np.array([0x00000801, 100], dtype='>i4').tobytes())
       
       # Write first 100 test labels as uint8
       labels = y_test[:100].astype(np.uint8)
       f.write(labels.tobytes())
   
   print("Created test data files for C inference")

# Export everything for C inference
export_weights_for_c(model, 'mnist_weights.bin')
create_test_files_for_c()

# Also save the standard format
model.save_weights('mnist.weights.h5')

print(f"Test accuracy: {model.evaluate(x_test, y_test)[1]:.3f}")
print("Ready for C inference!")
print("\nTo run C inference:")
print("gcc -g mnist_inference.c -o mnist_inference -lm")
print("./mnist_inference")