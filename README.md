# Neural Network-Based Handwritten Digit Recognition

> **Academic Project** for the subject **"Introduction to Artificial Intelligence" (College CA-2 Assignment)**

---

## 1. Introduction
Handwritten digit recognition is a classic problem in computer vision and artificial intelligence. Humans can easily read handwritten numbers regardless of handwriting styles, slants, or stroke thickness. However, for a computer, a handwritten image is merely a 2D matrix of numbers representing pixel intensity. 

This project demonstrates how a basic **Artificial Neural Network (ANN)** built with Python and TensorFlow/Keras can be trained to automatically recognize handwritten digits from **0 to 9** with high accuracy.

---

## 2. Problem Statement
Traditional rule-based programming cannot easily solve handwritten digit recognition because handwriting varies greatly from person to person. Writing thousands of `if-else` rules for pixel positions is impractical. 

Therefore, machine learning techniques—specifically Artificial Neural Networks—are required so that the machine can automatically learn visual feature patterns directly from training examples.

---

## 3. Objective
- Build and train a feed-forward Artificial Neural Network (ANN) using the benchmark MNIST dataset.
- Normalize image inputs and flatten 28x28 pixel images into 784 input values.
- Achieve high classification accuracy (>97%) on unseen test digits.
- Develop a beginner-friendly interactive web application using Streamlit where users can draw any digit from 0 to 9 on a canvas and view real-time predictions with confidence scores and probability distributions.

---

## 4. What is Artificial Neural Network (ANN)?
An **Artificial Neural Network (ANN)** is a computational model inspired by the structure and working of biological neurons in the human brain. It consists of interconnected nodes (neurons) organized into layers:

1. **Input Layer**: Receives raw features (in our project, 784 pixel values of an image).
2. **Hidden Layer(s)**: Processes the input by applying mathematical weights, biases, and non-linear activation functions (ReLU) to detect features like edges, curves, and loops.
3. **Output Layer**: Produces the final classification decision (probabilities for digits 0 through 9 using Softmax).

### General Flow Diagram:
```text
Input Layer (784 neurons) ──► Hidden Layer 1 (128 neurons) ──► Hidden Layer 2 (64 neurons) ──► Output Layer (10 neurons)
       │                                │                              │                             │
 (Pixel Values)                 (Feature Detection)            (Pattern Combination)            (Softmax Probabilities)
```

---

## 5. How the Neural Network Works
1. **Forward Propagation**:
   - Each input pixel \(x_i\) is multiplied by a weight \(w_i\), summed together with a bias \(b\):
     $$z = \sum (w_i \cdot x_i) + b$$
   - The result is passed through an activation function like **ReLU** (\(f(z) = \max(0, z)\)) to introduce non-linearity.
   - At the output layer, the **Softmax** function converts raw outputs into normalized probabilities that sum up to 1.0 (100%).

2. **Loss Calculation**:
   - The network compares its predicted probabilities with the actual true label using **Sparse Categorical Crossentropy** loss.

3. **Backpropagation & Optimization**:
   - The **Adam** optimizer calculates gradients and updates network weights to minimize the loss over training epochs.

4. **Final Prediction**:
   - The digit associated with the neuron having the highest Softmax probability is selected as the predicted class.

---

## 6. Dataset - MNIST
The **MNIST (Modified National Institute of Standards and Technology)** dataset is the benchmark dataset in Machine Learning for digit recognition.

- **Total Samples**: 70,000 grayscale images of handwritten digits.
- **Training Set**: 60,000 images.
- **Test Set**: 10,000 images.
- **Image Resolution**: 28 x 28 pixels (grayscale).
- **Labels**: Digits from `0` to `9`.

---

## 7. Data Preprocessing
To train the neural network effectively:
1. **Pixel Normalization**:
   - Original pixel values range from `0` (black) to `255` (white).
   - Pixel values are scaled to the range `[0.0, 1.0]` by dividing by 255.0. This prevents exploding gradients and speeds up model convergence.
2. **Flattening**:
   - Each 2D image matrix of size `28 x 28` is flattened into a 1D array of `784` continuous pixel values (`28 * 28 = 784`).

---

## 8. Model Architecture

| Layer Type | Layer Name | Input Shape | Output Neurons | Activation Function |
| :--- | :--- | :--- | :--- | :--- |
| **Input (Flatten)** | `input_flatten_784` | (28, 28) | 784 | None |
| **Hidden Layer 1** | `hidden_layer_128` | 784 | 128 | ReLU |
| **Hidden Layer 2** | `hidden_layer_64` | 128 | 64 | ReLU |
| **Output Layer** | `output_layer_10` | 64 | 10 | Softmax |

- **Total Parameters**: 109,386 trainable parameters.

---

## 9. Training
The model is compiled and trained with the following hyperparameters:

- **Optimizer**: Adam (Adaptive Moment Estimation)
- **Loss Function**: Sparse Categorical Crossentropy
- **Evaluation Metric**: Accuracy
- **Epochs**: 10
- **Batch Size**: 64
- **Validation Split**: 10% of training data

---

## 10. Testing and Evaluation
After training for 10 epochs, the model is evaluated on 10,000 unseen test images from the MNIST dataset:

- **Training Accuracy**: ~99.4%
- **Test Accuracy**: ~97.9%
- **Test Loss**: ~0.089

The training loss and accuracy curves are saved as a visual graph in `screenshots/training_history.png`.

---

## 11. Application Working
The project includes an interactive web application built with **Streamlit**:

1. **Drawing Canvas**: The user draws a digit on a 280x280 interactive canvas.
2. **Image Preprocessing Pipeline**:
   - Canvas image is captured as a grayscale image.
   - Resized down from 280x280 to 28x28 pixels using high-quality LANCZOS interpolation.
   - Scaled to range `[0.0, 1.0]`.
   - Reshaped to batch tensor `(1, 28, 28)`.
3. **Inference**:
   - The trained Keras model predicts output probabilities.
4. **Display**:
   - Displays the **Predicted Digit** (e.g. `Predicted Digit: 7`).
   - Displays **Confidence Percentage** (e.g. `Confidence: 97.45%`).
   - Displays an interactive **Bar Chart** showing probability distribution across all 10 digits (0–9).

---

## 12. Results
- The Artificial Neural Network achieves **over 97.5% accuracy** on test dataset images.
- Real-time digit predictions respond instantly when drawn on the Streamlit interface.
- Clear probability distribution helps students observe how confident the model is between ambiguous digits (e.g. distinguishing between `1` and `7` or `3` and `8`).

---

## 13. Advantages
- **Simple Architecture**: Easy to understand for AI beginners without complex convolutional layers.
- **Fast Training**: Model trains in less than 30 seconds on standard laptop CPUs.
- **Interactive UI**: Instant hands-on testing via drawing canvas.
- **Lightweight**: Minimum code footprint and small model file size (~430 KB).

---

## 14. Limitations
- **Position Dependency**: Standard dense neural networks expect centered digits; extreme offset drawings may decrease accuracy compared to Convolutional Neural Networks (CNNs).
- **Single Digit Only**: Designed for single digits (0–9), not multi-digit numbers.

---

## 15. Future Scope
- Upgrade the hidden architecture to a **Convolutional Neural Network (CNN)** for shift-invariant spatial recognition.
- Extend to multi-digit recognition using bounding box segmentation.
- Add support for recognizing handwritten English letters (EMNIST dataset).

---

## 16. Conclusion
This project successfully demonstrates the end-to-end workflow of an Artificial Neural Network for Handwritten Digit Recognition. From data preprocessing and model training to interactive prediction, it highlights fundamental AI/ML concepts in an educational and practical format suitable for college assignments.

---

## 17. Technologies Used
- **Python 3.11**
- **TensorFlow / Keras 3.x** - Deep Learning framework
- **NumPy** - Vectorized numerical processing
- **Matplotlib** - Performance plotting
- **Streamlit** - Web interface framework
- **Streamlit-Drawable-Canvas** - HTML5 drawing canvas integration
- **Pillow (PIL)** - Image resizing and processing

---

## 18. How to Run the Project

Follow these step-by-step commands to run the project on your computer:

### Step 1: Clone or Open Project Directory
Navigate to the project root directory in your terminal/command prompt:
```bash
cd "Neural Network-Based Handwritten Digit Recognition"
```

### Step 2: Create a Virtual Environment
```bash
python -m venv venv
```

### Step 3: Activate the Virtual Environment

- **Windows (Command Prompt / PowerShell)**:
```cmd
venv\Scripts\activate
```

- **Linux / macOS**:
```bash
source venv/bin/activate
```

### Step 4: Install Project Dependencies
```bash
pip install -r requirements.txt
```

### Step 5: Train the Neural Network Model
Run `train.py` to train the model, evaluate accuracy, and save `model/digit_model.keras`:
```bash
python train.py
```

### Step 6: Launch the Streamlit Web Application
```bash
streamlit run app.py
```
Open the local URL displayed in your terminal (typically `http://localhost:8501`) in any web browser to test the interactive drawing app!
