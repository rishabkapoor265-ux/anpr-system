import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import base64

from src.recognizer import LicensePlateRecognizer

# Page Configuration
st.set_page_config(page_title="ANPR Premium Dashboard", page_icon="🚗", layout="wide")

# Helper function to load CSS
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Helper function to get base64 of an image
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Inject Custom CSS
if os.path.exists("style.css"):
    load_css("style.css")

# --- HERO SECTION ---
logo_path = "assets/logo.png"
if os.path.exists(logo_path):
    logo_base64 = get_base64_image(logo_path)
    st.markdown(f'''
        <div class="logo-container">
            <img src="data:image/png;base64,{logo_base64}" class="logo-glow">
        </div>
    ''', unsafe_allow_html=True)

st.markdown('<h1 class="main-title">Automatic Number Plate Recognition</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Next-generation AI for vehicle detection and license plate extraction.</p>', unsafe_allow_html=True)

# --- SIDEBAR ---
st.sidebar.markdown('<h2 style="color: #D4AF37;">System Pipeline</h2>', unsafe_allow_html=True)
st.sidebar.markdown('''
<div class="glass-card" style="padding: 15px; background: rgba(212, 175, 55, 0.05);">
    <p>1. <b>Build Dataset:</b> Character generation</p>
    <p>2. <b>Train Model:</b> CNN Optimization</p>
    <p>3. <b>Inference:</b> Real-time recognition</p>
</div>
''', unsafe_allow_html=True)

@st.cache_resource
def load_recognizer():
    try:
        return LicensePlateRecognizer()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

recognizer = load_recognizer()

# --- MAIN CONTENT ---
with st.container():
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Drop your vehicle image here", type=["jpg", "jpeg", "png"])
    st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None and recognizer is not None:
    # Read the image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_image = cv2.imdecode(file_bytes, 1)
    
    st.markdown('<div class="result-header">Analysis Progress</div>', unsafe_allow_html=True)
    
    # Process the image
    with st.spinner("Decoding visuals..."):
        labeled_plate, plate_text, preprocessed_img, edged, plate_contour = recognizer.recognize(original_image)
    
    if plate_contour is not None:
        col1, col2 = st.columns([1, 1], gap="large")
        
        with col1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("🔍 Localization & Processing")
            
            # Sub-tabs for processing steps
            tab1, tab2 = st.tabs(["Edge Detection", "Plate Localization"])
            with tab1:
                st.image(edged, caption="Canny Edge Detection", use_column_width=True, channels="GRAY")
            with tab2:
                img_with_contour = preprocessed_img.copy()
                cv2.drawContours(img_with_contour, [plate_contour], -1, (0, 255, 0), 3)
                st.image(cv2.cvtColor(img_with_contour, cv2.COLOR_BGR2RGB), caption="Region of Interest (ROI)", use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("🎯 Extraction & Recognition")
            
            if isinstance(labeled_plate, np.ndarray):
                st.image(cv2.cvtColor(labeled_plate, cv2.COLOR_BGR2RGB), caption="Segmented Characters", use_column_width=True)
            
            st.write("---")
            st.markdown('<p style="color: #8892b0; margin-bottom: 0;">FINAL OUTPUT:</p>', unsafe_allow_html=True)
            st.markdown(f'<div class="detected-plate">{plate_text}</div>', unsafe_allow_html=True)
            st.success("Recognition Successful")
            st.markdown('</div>', unsafe_allow_html=True)
            
    else:
        st.error("No license plate detected. The model couldn't find a valid contour in the provided image.")
