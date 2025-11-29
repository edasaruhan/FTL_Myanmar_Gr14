import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# --- 1. System Settings ---
# Suppress TensorFlow warnings for cleaner logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- 2. Page Configuration ---
st.set_page_config(
    page_title="Waste Classification System",
    page_icon="♻️",
    layout="centered"
)

st.title("🔎 Waste Classification System")
st.write("Upload a single image to detect **Recyclability** AND **Material Type** simultaneously.")

# --- 3. Model Loading Functions ---
# We use distinct functions to load specific models from the same directory

@st.cache_resource
def load_binary_model():
    """Loads the Recyclable vs Non-Recyclable Model"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'binary_v2.h5')
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Error loading Binary Model: {e}")
        return None

@st.cache_resource
def load_type_model():
    """Loads the Material Type Model (Glass, Paper, etc.)"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'waste_type_model.h5')
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Error loading Type Model: {e}")
        return None

# Load both models at startup
with st.spinner('Loading AI Models...'):
    binary_model = load_binary_model()
    type_model = load_type_model()

# Stop execution if either model is missing
if binary_model is None or type_model is None:
    st.warning("⚠️ Please ensure both 'binary_v2.h5' and 'waste_type_model.h5' are in the directory.")
    st.stop()

# --- 4. Preprocessing Function ---
def process_image(image_data):
    """
    Prepares the image to match the training data format (224x224, Normalized).
    This function works for BOTH models.
    """
    size = (224, 224)
    image = ImageOps.fit(image_data, size, Image.LANCZOS)
    img_array = np.asarray(image)
    normalized_image_array = (img_array.astype(np.float32) / 255.0)
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    return data

# --- 5. User Interface & Logic ---
file = st.file_uploader("Choose a waste photo...", type=["jpg", "png", "jpeg"])

if file is not None:
    # A. Display Image
    image = Image.open(file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # B. Process Image (Once)
    processed_image = process_image(image)

    st.markdown("---")
    st.subheader("🤖 Analysis Results")

    # Create two columns for side-by-side results
    col1, col2 = st.columns(2)

    # --- MODEL 1: Binary Classification (Recyclable Status) ---
    with col1:
        st.markdown("### 1. Status")
        prediction_binary = binary_model.predict(processed_image)
        binary_labels = ['Non-Recycle', 'Recycle']
        
        bin_index = np.argmax(prediction_binary)
        bin_label = binary_labels[bin_index]
        bin_conf = np.max(prediction_binary)
        
        # Color coding based on result
        if bin_label == 'Recycle':
            st.success(f"**{bin_label}**")
        else:
            st.error(f"**{bin_label}**")
        st.write(f"Confidence: {bin_conf:.1%}")

    # --- MODEL 2: Multi-Class Classification (Material Type) ---
    with col2:
        st.markdown("### 2. Material")
        prediction_type = type_model.predict(processed_image)
        type_labels = ['Glass', 'Metal', 'Organic', 'Paper', 'Plastic']
        
        type_index = np.argmax(prediction_type)
        type_label = type_labels[type_index]
        type_conf = np.max(prediction_type)
        
        st.info(f"**{type_label}**")
        st.write(f"Confidence: {type_conf:.1%}")

    # --- Details Expander ---
    with st.expander("📊 See Full Probability Breakdown"):
        st.write("**Recyclability Probabilities:**")
        for i, label in enumerate(binary_labels):
            st.write(f"- {label}: {prediction_binary[0][i]:.1%}")
        
        st.write("**Material Type Probabilities:**")
        for i, label in enumerate(type_labels):
            st.write(f"- {label}: {prediction_type[0][i]:.1%}")