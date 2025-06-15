// mnist_inference.c - Complete MNIST inference in C
// Portable implementation for RISC-V profiling and acceleration
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define INPUT_SIZE 784   // 28x28 pixels
#define HIDDEN1_SIZE 128
#define HIDDEN2_SIZE 64  
#define OUTPUT_SIZE 10
#define NUM_TEST_IMAGES 100

// Global weight matrices (loaded from files)
static float weights1[INPUT_SIZE][HIDDEN1_SIZE];
static float weights2[HIDDEN1_SIZE][HIDDEN2_SIZE]; 
static float weights3[HIDDEN2_SIZE][OUTPUT_SIZE];
static float bias1[HIDDEN1_SIZE];
static float bias2[HIDDEN2_SIZE];
static float bias3[OUTPUT_SIZE];

// Test data
static float test_images[NUM_TEST_IMAGES][INPUT_SIZE];
static int test_labels[NUM_TEST_IMAGES];

// The function that will dominate our profiler output
float matrix_multiply_add(float input, float weight, float bias) {
    return (input * weight) + bias;  // This becomes 5 RISC-V instructions
}

// ReLU activation function
float relu(float x) {
    return x > 0 ? x : 0;
}

// Softmax activation for output layer
void softmax(float* input, float* output, int size) {
    float max_val = input[0];
    for (int i = 1; i < size; i++) {
        if (input[i] > max_val) max_val = input[i];
    }
    
    float sum = 0.0;
    for (int i = 0; i < size; i++) {
        output[i] = (float)exp((double)(input[i] - max_val));
        sum += output[i];
    }
    
    for (int i = 0; i < size; i++) {
        output[i] /= sum;
    }
}

// Load weights from binary file (exported from Python training)
int load_weights(const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error: Cannot open weights file %s\n", filename);
        return -1;
    }
    
    // Read weights in order: weights1, bias1, weights2, bias2, weights3, bias3
    fread(weights1, sizeof(float), INPUT_SIZE * HIDDEN1_SIZE, file);
    fread(bias1, sizeof(float), HIDDEN1_SIZE, file);
    fread(weights2, sizeof(float), HIDDEN1_SIZE * HIDDEN2_SIZE, file);
    fread(bias2, sizeof(float), HIDDEN2_SIZE, file);
    fread(weights3, sizeof(float), HIDDEN2_SIZE * OUTPUT_SIZE, file);
    fread(bias3, sizeof(float), OUTPUT_SIZE, file);
    
    fclose(file);
    printf("Loaded neural network weights successfully\n");
    return 0;
}

// Load test images and labels
int load_test_data(const char* images_file, const char* labels_file) {
    FILE* img_file = fopen(images_file, "rb");
    FILE* lbl_file = fopen(labels_file, "rb");
    
    if (!img_file || !lbl_file) {
        printf("Error: Cannot open test data files\n");
        return -1;
    }
    
    // Skip MNIST file headers (16 bytes for images, 8 bytes for labels)
    fseek(img_file, 16, SEEK_SET);
    fseek(lbl_file, 8, SEEK_SET);
    
    // Load test images and labels
    for (int i = 0; i < NUM_TEST_IMAGES; i++) {
        unsigned char pixels[INPUT_SIZE];
        unsigned char label;
        
        fread(pixels, sizeof(unsigned char), INPUT_SIZE, img_file);
        fread(&label, sizeof(unsigned char), 1, lbl_file);
        
        // Convert to float and normalize [0,1]
        for (int j = 0; j < INPUT_SIZE; j++) {
            test_images[i][j] = pixels[j] / 255.0f;
        }
        test_labels[i] = (int)label;
    }
    
    fclose(img_file);
    fclose(lbl_file);
    printf("Loaded %d test images successfully\n", NUM_TEST_IMAGES);
    return 0;
}

// The main inference function - where 89% of time is spent
int predict_digit(float pixels[INPUT_SIZE]) {
    float hidden1[HIDDEN1_SIZE];
    float hidden2[HIDDEN2_SIZE];
    float output_raw[OUTPUT_SIZE];
    float output[OUTPUT_SIZE];
    
    // Initialize with bias values
    memcpy(hidden1, bias1, HIDDEN1_SIZE * sizeof(float));
    memcpy(hidden2, bias2, HIDDEN2_SIZE * sizeof(float));
    memcpy(output_raw, bias3, OUTPUT_SIZE * sizeof(float));
    
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
            output_raw[j] += matrix_multiply_add(hidden2[i], weights3[i][j], 0);
        }
    }
    
    // Apply softmax to get probabilities
    softmax(output_raw, output, OUTPUT_SIZE);
    
    // Find predicted digit (argmax)
    int predicted = 0;
    for (int i = 1; i < OUTPUT_SIZE; i++) {
        if (output[i] > output[predicted]) predicted = i;
    }
    
    return predicted;  // Total: ~109,184 MAC operations per digit
}

// Benchmark inference performance (portable timing)
void benchmark_inference() {
    printf("\n=== MNIST Inference Benchmark ===\n");
    
    clock_t start, end;
    double total_time = 0.0;
    int correct = 0;
    
    // Warm-up run
    predict_digit(test_images[0]);
    
    // Benchmark loop
    start = clock();
    
    for (int i = 0; i < NUM_TEST_IMAGES; i++) {
        int prediction = predict_digit(test_images[i]);
        if (prediction == test_labels[i]) correct++;
    }
    
    end = clock();
    
    // Calculate timing (portable)
    total_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    double avg_time_ms = (total_time / NUM_TEST_IMAGES) * 1000.0;
    
    // Results
    printf("Processed %d images in %.3f seconds\n", NUM_TEST_IMAGES, total_time);
    printf("Average time per image: %.2f ms\n", avg_time_ms);
    printf("Throughput: %.1f images/second\n", NUM_TEST_IMAGES / total_time);
    printf("Accuracy: %d/%d (%.1f%%)\n", correct, NUM_TEST_IMAGES, 
           100.0 * correct / NUM_TEST_IMAGES);
    printf("Operations per image: ~109,184 MAC operations\n");
    printf("Total operations: ~%d million MACs\n", 
           (NUM_TEST_IMAGES * 109184) / 1000000);
}

int main(int argc, char* argv[]) {
    printf("MNIST Inference in C - Baseline Implementation\n");
    printf("============================================\n");
    
    // Load neural network weights
    if (load_weights("mnist_weights.bin") != 0) {
        return -1;
    }
    
    // Load test data
    if (load_test_data("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte") != 0) {
        return -1;
    }
    
    // Run benchmark
    benchmark_inference();
    
    // Single image test
    printf("\n=== Single Image Test ===\n");
    int prediction = predict_digit(test_images[0]);
    printf("First test image: predicted=%d, actual=%d %s\n", 
           prediction, test_labels[0], 
           (prediction == test_labels[0]) ? "✓" : "✗");
    
    return 0;
}