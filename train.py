import os
import matplotlib.pyplot as plt

# Flexible import for TensorFlow / Keras compatibility
try:
    import keras
    from keras import layers, models
    from keras.datasets import mnist
except ImportError:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models
    from tensorflow.keras.datasets import mnist

def main():
    print("=" * 60)
    print("   NEURAL NETWORK HANDWRITTEN DIGIT RECOGNITION (TRAINING)")
    print("=" * 60)

    # -------------------------------------------------------------
    # 1. Load MNIST Dataset
    # -------------------------------------------------------------
    print("\n[Step 1/6] Loading MNIST dataset...")
    # MNIST consists of 60,000 training images and 10,000 test images of 28x28 grayscale digits
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print(f"Loaded {len(x_train)} training samples and {len(x_test)} test samples.")

    # -------------------------------------------------------------
    # 2. Data Preprocessing
    # -------------------------------------------------------------
    print("\n[Step 2/6] Preprocessing data...")
    # Normalize pixel values from [0, 255] to [0.0, 1.0] for stable training
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # -------------------------------------------------------------
    # 3. Build Neural Network Architecture
    # -------------------------------------------------------------
    print("\n[Step 3/6] Building Feed-Forward Artificial Neural Network...")
    # Model architecture:
    # Input Layer: 784 neurons (28x28 flattened image)
    # Hidden Layer 1: 128 neurons, ReLU activation
    # Hidden Layer 2: 64 neurons, ReLU activation
    # Output Layer: 10 neurons, Softmax activation (probabilities for digits 0-9)
    model = models.Sequential([
        layers.Flatten(input_shape=(28, 28), name="input_flatten_784"),
        layers.Dense(128, activation='relu', name="hidden_layer_128"),
        layers.Dense(64, activation='relu', name="hidden_layer_64"),
        layers.Dense(10, activation='softmax', name="output_layer_10")
    ])

    model.summary()

    # -------------------------------------------------------------
    # 4. Compile the Model
    # -------------------------------------------------------------
    print("\n[Step 4/6] Compiling model...")
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # -------------------------------------------------------------
    # 5. Train the Neural Network
    # -------------------------------------------------------------
    epochs = 10
    print(f"\n[Step 5/6] Training neural network for {epochs} epochs...")
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=64,
        validation_split=0.1,
        verbose=1
    )

    # -------------------------------------------------------------
    # 6. Evaluate and Save Model
    # -------------------------------------------------------------
    print("\n[Step 6/6] Evaluating model on test dataset...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    train_accuracy = history.history['accuracy'][-1]

    print("\n" + "=" * 60)
    print("                  TRAINING RESULTS SUMMARY")
    print("=" * 60)
    print(f"Final Training Accuracy : {train_accuracy * 100:.2f}%")
    print(f"Test Accuracy           : {test_accuracy * 100:.2f}%")
    print(f"Test Loss               : {test_loss:.4f}")
    print("=" * 60)

    # Ensure output directories exist in project folder
    project_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(project_dir, "model")
    screenshots_dir = os.path.join(project_dir, "screenshots")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(screenshots_dir, exist_ok=True)

    # Save trained model
    model_path = os.path.join(model_dir, "digit_model.keras")
    model.save(model_path)
    print(f"\nModel saved successfully to: {model_path}")

    # Plot Training & Validation Loss and Accuracy graphs
    plt.figure(figsize=(12, 5))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange', linewidth=2)
    plt.title('Neural Network Accuracy', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss', color='blue', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', color='orange', linewidth=2)
    plt.title('Neural Network Loss', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    chart_path = os.path.join(screenshots_dir, "training_history.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"Training chart saved to: {chart_path}")
    print("\nTraining complete! You can now launch the Streamlit app using: streamlit run app.py\n")

if __name__ == "__main__":
    main()
