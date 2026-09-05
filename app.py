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
# Preprocessing Function (MNIST Bounding-Box Centering)
# -------------------------------------------------------------
def preprocess_drawn_image(img_rgba):
    """
    Preprocess user drawing to match official MNIST standards:
    1. Extract drawing bounding box (crop outer empty margins)
    2. Maintain aspect ratio & scale digit to fit inside 20x20 box
    3. Center the 20x20 digit inside a 28x28 black canvas (4-pixel border padding)
    4. Normalize pixel intensity values to [0.0, 1.0]
    """
    # Convert RGBA canvas array to PIL Grayscale Image
    img_pil = Image.fromarray(img_rgba.astype('uint8'), 'RGBA').convert('L')
    arr = np.array(img_pil)

    # Find non-zero drawn pixels (threshold > 10)
    coords = np.argwhere(arr > 10)
    if coords.size == 0:
        return None, None

    # Get bounding box coordinates
    min_y, min_x = coords.min(axis=0)
    max_y, max_x = coords.max(axis=0)

    # Crop digit using bounding box
    cropped = img_pil.crop((min_x, min_y, max_x + 1, max_y + 1))
    w, h = cropped.size

    # Scale cropped digit into 20x20 frame while preserving aspect ratio
    if w > h:
        new_w = 20
        new_h = max(1, int(round((h / w) * 20)))
    else:
        new_h = 20
        new_w = max(1, int(round((w / h) * 20)))

    resized_digit = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Paste 20x20 digit into the center of a black 28x28 canvas
    canvas_28x28 = Image.new("L", (28, 28), 0)
    paste_x = (28 - new_w) // 2
    paste_y = (28 - new_h) // 2
    canvas_28x28.paste(resized_digit, (paste_x, paste_y))

    # Normalize pixel intensity to range [0.0, 1.0]
    normalized = np.array(canvas_28x28, dtype=np.float32) / 255.0

    # Reshape tensor to (1, 28, 28) for Neural Network input
    input_tensor = np.reshape(normalized, (1, 28, 28))

    return input_tensor, canvas_28x28

# -------------------------------------------------------------
# Drawing Canvas Setup
# -------------------------------------------------------------
st.subheader("1. Draw a Digit (0-9)")
st.caption("Use your mouse or touch screen to draw a single digit inside the box below:")

# Canvas settings: 280x280 drawing area with 24px stroke width
canvas_result = st_canvas(
    fill_color="#000000",
    stroke_width=24,
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
        input_tensor, img_28x28 = preprocess_drawn_image(canvas_result.image_data)

        if input_tensor is None:
            st.warning("Please draw a digit on the canvas before clicking Predict!")
        else:
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

            # Display Centered 28x28 Input Preview
            st.write("**MNIST-Centered Preprocessed Image (28x28 pixels fed to Neural Network):**")
            st.image(img_28x28, width=140, caption="Auto-centered 28x28 MNIST input")

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
