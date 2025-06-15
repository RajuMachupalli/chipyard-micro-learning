// mnist_inference.c - The code we'll profile and accelerate
#include <stdio.h>
#include <math.h>

#define INPUT_SIZE 784   // 28x28 pixels
#define HIDDEN1_SIZE 128
#define HIDDEN2_SIZE 64  
#define OUTPUT_SIZE 10

// The function that will dominate our profiler output
float matrix_multiply_add(float input, float weight, float bias) {
    return (input * weight) + bias;  // This becomes 5 RISC-V instructions
}

// ReLU activation function
float relu(float x) {
    return x > 0 ? x : 0;
}

// The main inference function - where 89% of time is spent
int predict_digit(float pixels[INPUT_SIZE], 
                  float weights1[INPUT_SIZE][HIDDEN1_SIZE],
                  float weights2[HIDDEN1_SIZE][HIDDEN2_SIZE],
                  float weights3[HIDDEN2_SIZE][OUTPUT_SIZE]) {
    
    float hidden1[HIDDEN1_SIZE] = {0};
    float hidden2[HIDDEN2_SIZE] = {0};
    float output[OUTPUT_SIZE] = {0};
    
    // Layer 1: 784 × 128 = 100,352 MAC operations
    for (int i = 0; i < INPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN1_SIZE; j++) {
            hidden1[j] += matrix_multiply_add(pixels[i], weights1[i][j], 0);
        }
    }
    for (int j = 0; j < HIDDEN1_SIZE; j++) hidden1[j] = relu(hidden1[j]);
    
    // Layer 2: 128 × 64 = 8,192 MAC operations  
    for (int i = 0; i < HIDDEN1_SIZE; i++) {
        for (int j = 0; j < HIDDEN2_SIZE; j++) {
            hidden2[j] += matrix_multiply_add(hidden1[i], weights2[i][j], 0);
        }
    }
    for (int j = 0; j < HIDDEN2_SIZE; j++) hidden2[j] = relu(hidden2[j]);
    
    // Layer 3: 64 × 10 = 640 MAC operations
    for (int i = 0; i < HIDDEN2_SIZE; i++) {
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            output[j] += matrix_multiply_add(hidden2[i], weights3[i][j], 0);
        }
    }
    
    // Find predicted digit (argmax)
    int predicted = 0;
    for (int i = 1; i < OUTPUT_SIZE; i++) {
        if (output[i] > output[predicted]) predicted = i;
    }
    
    return predicted;  // Total: ~109,184 MAC operations per digit
}