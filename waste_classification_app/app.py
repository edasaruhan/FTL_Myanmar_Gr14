#updated ver
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
    page_title="Waste Classification System (Decision-Support)",
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

# --- 4. Helper Functions for New Features ---

def get_sorting_instructions(material_type, is_recyclable):
    """Generates the Sorting Instruction Card based on results."""
    # Define instructions for the decision-support tool
    instructions = {
        ('Paper', 'Recycle'): "✅ **RECYCLE:** Flatten the cardboard/paper. Ensure it is clean and dry. NO pizza boxes or greasy materials.",
        ('Paper', 'Non-Recycle'): "🗑️ **DISPOSE:** This is likely contaminated or non-recyclable paper (e.g., thermal paper, shredded paper). Place in general waste.",
        ('Plastic', 'Recycle'): "✅ **RECYCLE:** Rinse the container (remove food residue). Look for a recycling symbol 1, 2, or 5. If it's film/bag, check local rules.",
        ('Plastic', 'Non-Recycle'): "🗑️ **DISPOSE:** This plastic is likely single-use, non-identifiable, or too dirty. Place in general waste.",
        ('Glass', 'Recycle'): "✅ **RECYCLE:** Rinse bottles/jars (remove caps). Do NOT include broken glass or window glass.",
        ('Metal', 'Recycle'): "✅ **RECYCLE:** Rinse cans/tins (remove residue). Flatten if possible. Ensure no sharp edges.",
        ('Metal', 'Non-Recycle'): "🗑️ **DISPOSE:** Small metal objects or mixed material items (e.g., batteries - which should be treated as Hazardous Waste separately).",
        ('Organic', 'Recycle'): "♻️ **COMPOST:** Place food scraps, yard waste, or compostable materials into a dedicated compost bin.",
        ('Organic', 'Non-Recycle'): "🗑️ **DISPOSE:** If this is heavily processed or non-compostable organic material (e.g., treated wood), place in general waste."
    }
    
    status = 'Recycle' if is_recyclable else 'Non-Recycle'
    return instructions.get((material_type, status), f"❓ **UNCERTAIN:** Material Type: {material_type}, Status: {status}. Check local waste guidelines.")

def generate_quality_flags(image_data):
    """
    Simulates simple image quality checks based on basic image properties.
    In a real system, this would use a separate model or sophisticated OpenCV analysis.
    """
    img_array = np.asarray(image_data)
    
    # Simple check for 'Low Lighting' (based on average pixel intensity)
    avg_brightness = np.mean(img_array)
    is_low_lighting = avg_brightness < 80 # Threshold for very dark images (0-255 scale)
    
    # Placeholder for 'Background Clutter' (more complex, using edge detection or feature count)
    # We will assume a random chance for demonstration, or a simple placeholder.
    is_clutter = np.random.choice([True, False], p=[0.2, 0.8]) # 20% chance of clutter flag
    
    flags = []
    if is_low_lighting:
        flags.append("💡 **Low Lighting Detected** (May impact color-based prediction)")
    if is_clutter:
        flags.append("🖼️ **Background Clutter Detected** (Model might focus on incorrect area)")
        
    return flags

# --- 5. Preprocessing Function ---
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

# --- 6. User Interface & Logic ---
file = st.file_uploader("Choose a waste photo...", type=["jpg", "png", "jpeg"])

if file is not None:
    # A. Display Image
    image = Image.open(file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # B. Process Image (Once)
    processed_image = process_image(image)
    
    # C. Generate Quality Flags (New Feature 4)
    quality_flags = generate_quality_flags(image)

    st.markdown("---")
    st.subheader("🤖 Analysis Results")

    # Create two columns for side-by-side results
    col1, col2 = st.columns(2)

    # --- MODEL 1: Binary Classification (Recyclable Status) ---
    with col1:
        st.markdown("### 1. Status")
        prediction_binary = binary_model.predict(processed_image, verbose=0)
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
        prediction_type = type_model.predict(processed_image, verbose=0)
        type_labels = ['Glass', 'Metal', 'Organic', 'Paper', 'Plastic']
        
        type_index = np.argmax(prediction_type)
        type_label = type_labels[type_index]
        type_conf = np.max(prediction_type)
        
        st.info(f"**{type_label}**")
        st.write(f"Confidence: {type_conf:.1%}")
        
    st.markdown("---")
    
    # --- Decision Support Panel (New Feature 1 & 2) ---
    
    st.subheader("💡 Decision Support & Sorting Guidance")
    
    # 1. Confidence-based explanation panel
    if bin_conf < 0.7 or type_conf < 0.7:
        st.warning("⚠️ **Low Confidence Flag:** One or both model predictions are below 70%. Be cautious with the recommendation.")
        st.write("The model might be **'confused'** because the image has features of multiple classes (e.g., plastic bottle with a metal lid, or a wet paper bag).")
        st.write("Check the full probability breakdown below for competing classes.")
    
    # 2. Sorting Instruction Card Generator
    is_recyclable = (bin_label == 'Recycle')
    instructions = get_sorting_instructions(type_label, is_recyclable)
    
    st.markdown(f"""
    <div style="padding: 15px; border-radius: 10px; border: 2px solid #007BFF; background-color: #F8F9FA;">
        <h4 style="margin-top: 0; color: #007BFF;">♻️ Sorting Instruction Card (SDG 12)</h4>
        <p style="font-size: 1.1em; margin-bottom: 0;color:black;">{instructions}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # --- Quality Flags (New Feature 4) ---
    # if quality_flags:
    #     st.subheader("🚨 Input Quality Review")
    #     for flag in quality_flags:
    #         st.code(flag)
    # else:
    #     st.success("✅ Input Image Quality is high.")
        
    # st.markdown("---")

    # --- Details Expander (Original + New Feature 3 Placeholder) ---
    with st.expander("📊 Full Technical Breakdown & Confusion Summary"):
        
        # 3. Visual Confusion Summary (Placeholder - requires static data)
        st.markdown("#### Visual Confusion Summary (Based on Testing Data)")
        st.write("""
        * **Plastic** is commonly confused with **Paper** (if crumpled and white).
        * **Metal** is sometimes confused with **Glass** (due to reflections/shine).
         * **Plastic** is sometimes confused with **Glass** (transparent or glossy plastics).
        """)
        
        st.markdown("#### Raw Prediction Probabilities")
        
        st.write("**Recyclability Probabilities:**")
        for i, label in enumerate(binary_labels):
            st.write(f"- {label}: {prediction_binary[0][i]:.1%}")
        
        st.write("**Material Type Probabilities:**")
        for i, label in enumerate(type_labels):
            st.write(f"- {label}: {prediction_type[0][i]:.1%}")