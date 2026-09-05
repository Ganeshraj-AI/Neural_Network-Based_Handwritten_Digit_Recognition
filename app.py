import os
import numpy as np
import pandas as pd
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
# 1. Streamlit App Page Configuration (Academic Layout)
# -------------------------------------------------------------
st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------
# 2. Sidebar - Academic Project Information
# -------------------------------------------------------------
with st.sidebar:
    st.header("🎓 Academic Information")
    st.markdown("**Subject:** Introduction to Artificial Intelligence")
    st.markdown("**Project:** Handwritten Digit Recognition")
    st.markdown("---")
    
    st.subheader("🧠 Neural Network Architecture")
    st.markdown("""
    - **Input Layer:** 784 neurons (28×28 pixels)
    - **Hidden Layer 1:** 128 neurons (ReLU)
    - **Hidden Layer 2:** 64 neurons (ReLU)
    - **Output Layer:** 10 neurons (Softmax)
    """)
    st.markdown("---")
    
    st.subheader("⚙️ Training Hyperparameters")
    st.markdown("""
    - **Dataset:** MNIST (60,000 train / 10,000 test)
    - **Optimizer:** Adam
    - **Loss:** Sparse Categorical Crossentropy
    - **Test Accuracy:** ~97.9%
    """)

# -------------------------------------------------------------
# 3. Main Header & Description
# -------------------------------------------------------------
st.title("Handwritten Digit Recognition using Neural Network")
st.markdown("Draw a handwritten digit from **0 to 9** on the canvas below and let the artificial neural network predict it in real-time.")
st.markdown("---")

# -------------------------------------------------------------
# 4. Model Verification & Loading
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
        "Please run `python train.py` first to train and save the model."
    )
    st.stop()

model = get_model(MODEL_PATH)

# -------------------------------------------------------------
# 5. MNIST Bounding-Box Centering Preprocessor
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
# 6. Main UI Columns Setup
# -------------------------------------------------------------
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("1. Interactive Drawing Canvas")
    st.caption("Draw a single digit (0-9) inside the box below:")

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

    predict_btn = st.button("🔍 Predict Digit", type="primary", use_container_width=True)

with col2:
    st.subheader("2. Prediction & Probability Analysis")

    if predict_btn:
        if canvas_result.image_data is not None:
            input_tensor, img_28x28 = preprocess_drawn_image(canvas_result.image_data)

            if input_tensor is None:
                st.warning("⚠️ Please draw a digit on the canvas before clicking Predict!")
            else:
                # Perform Prediction using Artificial Neural Network
                raw_predictions = model.predict(input_tensor, verbose=0)
                predictions = raw_predictions[0]
                predicted_digit = int(np.argmax(predictions))
                confidence = float(np.max(predictions)) * 100.0

                # Display Results in Metric Cards
                m_col1, m_col2 = st.columns(2)
                with m_col1:
                    st.metric(label="Predicted Digit", value=f"Digit {predicted_digit}")
                with m_col2:
                    st.metric(label="Confidence Score", value=f"{confidence:.2f}%")

                st.markdown("---")
                
                # Display 28x28 Preprocessed Input Image
                img_col1, img_col2 = st.columns([1, 2])
                with img_col1:
                    st.write("**MNIST Centered Input (28×28):**")
                    st.image(img_28x28, width=110, caption="Auto-Centered 28x28")
                with img_col2:
                    st.write("**Preprocessing Info:**")
                    st.caption("• Cropped stroke bounding box")
                    st.caption("• Rescaled into 20×20 preserving aspect ratio")
                    st.caption("• Centered inside 28×28 black frame")

                st.markdown("---")
                st.write("**Softmax Output Probabilities (Digits 0-9):**")

                # Prepare DataFrame for chart & table
                df_probs = pd.DataFrame({
                    "Digit": [f"Digit {i}" for i in range(10)],
                    "Probability": [float(p) for p in predictions]
                }).set_index("Digit")

                st.bar_chart(df_probs, height=220)

                # Detailed numerical table
                st.dataframe(
                    pd.DataFrame({
                        "Digit": list(range(10)),
                        "Probability": [f"{predictions[i]:.4f}" for i in range(10)],
                        "Percentage": [f"{predictions[i]*100:.2f}%" for i in range(10)]
                    }),
                    use_container_width=True,
                    hide_index=True
                )
    else:
        st.info("👈 Draw a digit on the left canvas and click **Predict Digit** to view results.")
