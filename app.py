import os
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Flexible Keras loading for TensorFlow / Keras 3.x compatibility
try:
    import keras
    load_model_fn = keras.models.load_model
except ImportError:
    import tensorflow as tf
    load_model_fn = tf.keras.models.load_model

# -------------------------------------------------------------
# Streamlit App Page Configuration (Basic Academic Layout)
# -------------------------------------------------------------
st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="🔢",
    layout="centered"
)

# -------------------------------------------------------------
# Title and Description
# -------------------------------------------------------------
st.title("Handwritten Digit Recognition using Neural Network")
st.write("Draw a handwritten digit from 0 to 9 and let the neural network predict it.")
st.markdown("---")

# -------------------------------------------------------------
# Model Verification & Loading
# -------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "digit_model.keras")

@st.cache_resource
def get_model(path):
    return load_model_fn(path)

if not os.path.exists(MODEL_PATH):
    st.error(
        "⚠️ **Model File Not Found!**\n\n"
        f"The trained model file `{MODEL_PATH}` does not exist.\n\n"
        "Please execute the training script first:\n"
        "```bash\npython train.py\n```"
    )
    st.stop()

model = get_model(MODEL_PATH)

# -------------------------------------------------------------
# Drawing Canvas Setup
# -------------------------------------------------------------
st.subheader("1. Draw a Digit (0-9)")
st.caption("Use your mouse or touch screen to draw a single digit inside the box below:")

# Canvas settings: 280x280 drawing area (10x scale of 28x28)
canvas_result = st_canvas(
    fill_color="#000000",
    stroke_width=20,
    stroke_color="#FFFFFF",      # White stroke on black background (MNIST format)
    background_color="#000000",  # Black background
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

st.write("Click the button below to analyze your drawing.")
predict_btn = st.button("Predict Digit", type="primary")

# -------------------------------------------------------------
# Preprocessing and Prediction Logic
# -------------------------------------------------------------
if predict_btn:
    if canvas_result.image_data is not None:
        # Extract pixel data from canvas (RGBA numpy array)
        img_array = canvas_result.image_data

        # Check if user has drawn anything (sum of RGB channels > 0)
        if np.sum(img_array[:, :, :3]) == 0:
            st.warning("Please draw a digit on the canvas before clicking Predict!")
        else:
            # Convert RGBA numpy array to PIL Image
            img = Image.fromarray(img_array.astype('uint8'), 'RGBA')
            
            # Convert image to Grayscale ('L')
            img_gray = img.convert('L')

            # Resize image from 280x280 down to 28x28 using High-quality LANCZOS resampling
            img_resized = img_gray.resize((28, 28), Image.Resampling.LANCZOS)

            # Convert PIL image to numpy array of float32 and normalize pixel values to [0.0, 1.0]
            img_normalized = np.array(img_resized, dtype=np.float32) / 255.0

            # Reshape image to batch shape (1, 28, 28) for neural network input
            input_tensor = np.reshape(img_normalized, (1, 28, 28))

            # Perform Prediction using Artificial Neural Network
            raw_predictions = model.predict(input_tensor, verbose=0)
            predictions = raw_predictions[0]
            predicted_digit = int(np.argmax(predictions))
            confidence = float(np.max(predictions)) * 100.0

            st.markdown("---")
            st.subheader("2. Prediction Result")

            # Display Predicted Digit and Confidence
            st.markdown(f"### Predicted Digit: **{predicted_digit}**")
            st.markdown(f"### Confidence: **{confidence:.2f}%**")

            # Display Processed 28x28 Input Preview
            st.write("**Processed Input Image (28x28 pixels as fed into Neural Network):**")
            st.image(img_resized, width=140, caption="Normalized 28x28 MNIST input")

            st.markdown("---")
            st.subheader("3. Probability Distribution across Digits (0-9)")
            st.write("The output layer uses the **Softmax** activation function to output probabilities for each class:")

            # Prepare data for probability table and bar chart
            prob_dict = {f"Digit {i}": float(predictions[i]) for i in range(10)}
            
            # Display Bar Chart of Probabilities
            st.bar_chart(prob_dict)

            # Display detailed numerical table
            st.write("**Detailed Probabilities Table:**")
            st.dataframe(
                [{"Digit": i, "Probability": f"{predictions[i]:.4f}", "Percentage": f"{predictions[i]*100:.2f}%"} for i in range(10)],
                use_container_width=True
            )
else:
    st.info("Draw a digit above and click 'Predict Digit'.")
