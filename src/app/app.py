import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import os
import time
import glob
import pickle
import sys

# Add project root to path for config access if needed
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import IMU generator for manual uploads
from src.data_acquisition import generate_imu

MODEL_PATH = "models/optimized_model.h5"  # Loaded optimized model from Stage 6
SCALER_PATH = "config/preprocessing_params.pkl"
DATA_DIR = os.path.join(project_root, "data", "test") # Use test data for demo
CLASS_NAMES = ['asphalt', 'carpet', 'concrete', 'grass', 'tile']

st.set_page_config(
    page_title="Surface Classifier",
    layout="wide"
)

# Custom class to fix Dense layer deserialization issue (quantization_config)
@tf.keras.utils.register_keras_serializable()
class FixedDense(tf.keras.layers.Dense):
    def __init__(self, *args, **kwargs):
        # Strip unexpected arguments that might be present in newer config versions
        if 'quantization_config' in kwargs:
            kwargs.pop('quantization_config')
        super().__init__(*args, **kwargs)

@st.cache_resource
def load_resources():
    """Loads the NN model and Scaler."""
    resources = {}
    
    # Load Model with custom objects to handle deserialization errors
    if os.path.exists(MODEL_PATH):
        try:
            # Attempt standard load first
            resources['model'] = tf.keras.models.load_model(MODEL_PATH)
        except Exception as e1:
            # Fallback to custom objects if 'quantization_config' error occurs
            try:
                # st.warning(f"Standard load failed, applying Dense layer fix...") # Debug info
                resources['model'] = tf.keras.models.load_model(
                    MODEL_PATH, 
                    custom_objects={'Dense': FixedDense}
                )
                # st.success("Model loaded with compatibility fix.")
            except Exception as e2:
                st.error(f"Error loading model (Standard: {e1}) (Fixed: {e2})")
                return None
    else:
        st.error(f"Model not found at {MODEL_PATH}")
        return None

    # Load Scaler
    if os.path.exists(SCALER_PATH):
        with open(SCALER_PATH, 'rb') as f:
            resources['scaler'] = pickle.load(f)
    else:
        st.warning(f"Scaler not found at {SCALER_PATH}. Using raw IMU data.")
        resources['scaler'] = None
        
    return resources

@st.cache_resource
def load_imu_stats(processed_dir):
    """Loads IMU statistics for synthetic generation."""
    if not os.path.exists(processed_dir):
        return None
    # Suppress print during stats calculation if possible, or just let it run
    return generate_imu.calculate_stats(processed_dir)

def handle_manual_file(uploaded_file, stats):
    """Saves uploaded image and generates matching IMU data based on filename."""
    manual_dir = os.path.join(project_root, "data", "manual")
    if not os.path.exists(manual_dir):
        os.makedirs(manual_dir)
        
    # Save Uploaded Image - overwrites previous manual upload
    # Using specific name to identify it easily or generic "manual_img.jpg"
    img_path = os.path.join(manual_dir, "manual_img.jpg")
    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    # Infer Class from Filename
    filename = uploaded_file.name.lower()
    inferred_label = "unknown"
    for cls in CLASS_NAMES:
        if cls in filename:
            inferred_label = cls
            break
            
    # Generate Synthetic IMU
    imu_path = os.path.join(manual_dir, "manual_imu.npy")
    
    if inferred_label in stats:
        mean = stats[inferred_label]['mean']
        std = stats[inferred_label]['std']
        # Generate with same shape as mean
        noise = np.random.normal(0, 1, mean.shape)
        generated_imu = mean + (noise * std)
        np.save(imu_path, generated_imu)
        st.toast(f"Synthesized IMU data for inferred class: {inferred_label}")
    else:
        # Fallback to Concrete or Zero if unknown
        st.warning(f"Class could not be inferred from filename '{filename}'. Using 'concrete' profile for IMU.")
        if 'concrete' in stats:
             mean = stats['concrete']['mean']
             std = stats['concrete']['std']
             generated_imu = mean + (np.random.normal(0, 1, mean.shape) * std)
             np.save(imu_path, generated_imu)
        else:
             st.error("Stats not available for fallback. Using zeros.")
             np.save(imu_path, np.zeros((99, 10)))

    return imu_path, img_path, inferred_label

def get_random_test_sample():
    """Picks a random sample from the test set."""
    if not os.path.exists(DATA_DIR):
        return None, None, "Data directory not found"
        
    # Get all IMU files
    all_imu_files = []
    for cls in CLASS_NAMES:
        cls_dir = os.path.join(DATA_DIR, cls)
        if os.path.exists(cls_dir):
            files = glob.glob(os.path.join(cls_dir, "*_imu.npy"))
            all_imu_files.extend([(f, cls) for f in files])
            
    if not all_imu_files:
        return None, None, "No files found in test set"
        
    # Pick random
    imu_path, true_label = all_imu_files[np.random.randint(len(all_imu_files))]
    base_name = imu_path.replace("_imu.npy", "")
    img_path = f"{base_name}_img.jpg"
    
    if not os.path.exists(img_path):
        # Fallback to just loading whatever
        return get_random_test_sample() # Try again
    return imu_path, img_path, true_label

def preprocess_sample(imu_path, img_path, scaler):
    # Load IMU
    imu_data = np.load(imu_path)
    # Ensure shape (99, 10)
    if imu_data.shape != (99, 10):
        # Allow basic reshape if possible
        if imu_data.size == 990:
             imu_data = imu_data.reshape(99, 10)
    
    # Scale IMU
    if scaler:
        imu_data = scaler.transform(imu_data)
        
    # Add batch dim -> (1, 99, 10)
    imu_input = np.expand_dims(imu_data, axis=0)
    
    # Load Image
    img = tf.keras.utils.load_img(img_path, color_mode='grayscale', target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = img_array / 255.0
    img_input = np.expand_dims(img_array, axis=0)
    
    return imu_input, img_input

def main():
    st.title("Surface Classification - Inference")
    st.markdown("Trained Model Inference")
    
    resources = load_resources()
    if not resources:
        st.stop()
        
    model = resources['model']
    scaler = resources['scaler']

    st.sidebar.header("Control Panel")
    
    # Mode Selection
    mode = st.sidebar.radio("Input Source", ["Random Test Sample", "Manual Upload"])
    
    imu_path, img_path, true_label = None, None, None
    run_inference = False
    
    if mode == "Random Test Sample":
        if st.sidebar.button("Load Random Test Sample"):
            with st.spinner("Loading random sample..."):
                imu_path, img_path, true_label = get_random_test_sample()
                run_inference = True
    else:
        st.sidebar.markdown("---")
        st.sidebar.info("Filename must contain class name (e.g. 'concrete_01.jpg') to generate matching IMU.")
        uploaded_file = st.sidebar.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])
        
        # Pre-load stats for logic
        processed_dir = os.path.join(project_root, "data", "processed")
        stats = load_imu_stats(processed_dir)
        
        if uploaded_file:
            if stats:
                with st.spinner("Processing Upload & Synthesizing IMU..."):
                    imu_path, img_path, true_label = handle_manual_file(uploaded_file, stats)
                    run_inference = True
            else:
                 st.error("Could not load IMU stats (data/processed missing?). Cannot generate matching IMU.")

    # Execution Block
    if run_inference and imu_path:
        st.success(f"Loaded: {os.path.basename(img_path)}") 
        if mode == "Manual Upload":
             st.info(f"Inferred Class from filename: **{true_label}** (Synthetic IMU generated)")
        else:
             st.info(f"True Label: **{true_label}**")
            
        # Preprocess
        input_imu, input_img = preprocess_sample(imu_path, img_path, scaler)
        
        # Visualization
        col_viz1, col_viz2, col_viz3 = st.columns(3)
        with col_viz1:
            st.markdown("**IMU Signal (Scaled)**")
            st.line_chart(pd.DataFrame(input_imu[0], columns=[f"Ax{i}" for i in range(10)]))
        
        with col_viz2:
            st.markdown("**Original Image**")
            st.image(img_path, caption=f"Source: {true_label}")

        with col_viz3:
            st.markdown("**Preprocessed Input**")
            # input_img is (1, 224, 224, 1), normalized 0-1
            st.image(input_img[0], caption="Grayscale 224x224 (0-1)", clamp=True)
        
        # Inference
        start_time = time.time()
        prediction = model.predict([input_imu, input_img])
        inf_time = (time.time() - start_time) * 1000
        
        predicted_idx = np.argmax(prediction)
        predicted_class = CLASS_NAMES[predicted_idx]
        confidence = prediction[0][predicted_idx] * 100
        
        # Results
        st.markdown("---")
        st.subheader("Classification Results")
        
        # --- STATE MACHINE LOGIC: THRESHOLD CHECK ---
        CONFIDENCE_THRESHOLD = 70.0  # Defined in State Machine logic
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted", predicted_class, delta=None)
        
        # Display Confidence with visual indicator
        if confidence < CONFIDENCE_THRESHOLD:
            c2.metric("Confidence", f"{confidence:.1f}%", delta="-Low Confidence", delta_color="inverse")
        else:
            c2.metric("Confidence", f"{confidence:.1f}%", delta="High Confidence")
            
        c3.metric("Inference Time", f"{inf_time:.1f} ms")
        
        # --- STATE MACHINE LOGIC: ACT vs LOG ---
        if confidence < CONFIDENCE_THRESHOLD:
            st.warning(f" **UNCERTAIN PREDICTION (<{CONFIDENCE_THRESHOLD}%)**.")
        else:
        
            # if label is 'unknown' show result
            if true_label != "unknown" and predicted_class == true_label:
                st.success(f" **CONFIRMED ({confidence:.1f}%)**.`INFERENCE` → `ACT`")
            elif true_label != "unknown" and predicted_class != true_label:
                if mode == "Manual Upload":
                     st.warning(f" Prediction ({predicted_class}) differs from filename target ({true_label})")
                else:
                     st.error(f" Prediction Incorrect. Expected {true_label}")
            else:
                 st.success(f"Prediction: **{predicted_class}**")
        
        # Probabilities
        st.bar_chart(pd.DataFrame(prediction.T, index=CLASS_NAMES, columns=["Probability"]))
            
    elif run_inference and not imu_path:
         st.warning("Could not load sample.")
    
    else:
        if mode == "Random Test Sample":
            st.write("Click the button in the sidebar to load and classify a random sample from the test set.")

if __name__ == "__main__":
    main()
