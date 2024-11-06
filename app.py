import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from ultralytics import YOLO
import cv2
from collections import Counter
from tensorflow.keras import backend as K
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras.preprocessing import image

@register_keras_serializable()
def rgb_to_grayscale(input):
    """Average out each pixel across its 3 RGB layers resulting in a grayscale image."""
    return K.mean(input, axis=3)

@register_keras_serializable()
def rgb_to_grayscale_output_shape(input_shape):
    return input_shape[:-1]

# Load models
fingerprint_model = load_model('fingerprint.h5')
bloodsample_model = load_model('bloodsample.h5')
bloodcellcount_model = YOLO('best.pt')
celltype_model = load_model('bloodcelltype1.keras')
nuclear_model = load_model('bloodcelltype2.keras')

# Class dictionaries for cell types and nuclear types
class1 = {1: 'NEUTROPHIL', 2: 'EOSINOPHIL', 3: 'MONOCYTE', 4: 'LYMPHOCYTE'}
class2 = {0: 'Mononuclear', 1: 'Polynuclear'}
blood_group_labels = ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']

# Set the app title, layout, and favicon
st.set_page_config(page_title="BloodIQ", layout="wide", page_icon="favicon.ico")

# Custom CSS for styling, including a background image
def get_css(button_color, hover_color):
    return f"""
    <style>
    body {{
        background-image: url('https://images.unsplash.com/photo-1601091689021-6e1f65a7d5f1');
        background-size: cover;
        background-position: center;
        font-family: 'Arial', sans-serif;
        color: #ffffff;
    }}
    .header {{
        text-align: center;
        padding: 30px;
        font-size: 36px;
        font-weight: bold;
    }}
    .button {{
        margin: 10px 0;
        padding: 15px 30px;
        background-color: {button_color};
        color: white;
        border: none;
        border-radius: 5px;
        font-size: 18px;
        font-weight: bold;
        transition: background-color 0.3s ease;
        width: 100%;
    }}
    .button:hover {{
        background-color: {hover_color};
    }}
    .section-header {{
        color: #ffffff;
        font-size: 28px;
        margin-top: 40px;
        text-align: center;
    }}
    .output {{
        background-color: rgba(255, 255, 255, 0.8);
        border-radius: 5px;
        padding: 20px;
        margin: 20px;
        text-align: center;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
    }}
    </style>
    """

# Default button colors
button_color = "#007bff"
hover_color = "#0056b3"
st.markdown(get_css(button_color, hover_color), unsafe_allow_html=True)

# Header Section
st.markdown("<h1 class='header'>Integrated System for Blood Group Detection and Hematological Analysis Using Fingerprint and Blood Sample Models</h1>", unsafe_allow_html=True)
st.write("This app allows you to detect blood groups and perform hematological analysis using fingerprint and blood sample models.")

# Helper functions for blood group detection
def predict_blood_group(img, model):
    img = img.resize((224, 224))  # Resize to match the model's input shape
    image_array = image.img_to_array(img)
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    image_array = image_array / 255.0  # Normalize to match training data

    predictions = model.predict(image_array)
    predicted_class = np.argmax(predictions, axis=1)
    return predicted_class[0]

# Blood Cell Counting and Subtype Prediction
def analyze_blood_sample(image):
    # Predict cell count using bloodcellcount model
    result = bloodcellcount_model.predict(source=image, imgsz=640, conf=0.25)

    # Annotate image
    annotated_img = result[0].plot()
    annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)  # Convert color for Streamlit

    # Get class counts
    detections = result[0].boxes.data
    class_names = [bloodcellcount_model.names[int(cls)] for cls in detections[:, 5]]
    count = Counter(class_names)

    # Display detection counts
    detection_str = ', '.join([f"{name}: {count}" for name, count in count.items()])

    # Convert BGR to RGB for Streamlit
    annotated_img_rgb = annotated_img[:, :, ::-1]

    # Predict cell type and nuclear type
    img_small = image.resize((80, 60))  # Resize to the input size of subtype models
    img_array_small = np.array(img_small) / 255.0
    img_array_small = np.expand_dims(img_array_small, axis=0)

    celltype_pred = celltype_model.predict(img_array_small)
    nuclear_pred = nuclear_model.predict(img_array_small)

    celltype_label = class1[np.argmax(celltype_pred)]
    nuclear_label = class2[np.argmax(nuclear_pred)]

    # Display annotated image with bounding boxes
    st.image(annotated_img_rgb, caption="Annotated Blood Sample Image", width = 400)
    # Return results as a dictionary
    return {
        "Cell Counts": detection_str,
        "Cell Type": celltype_label,
        "Nuclear Type": nuclear_label
    }

# Section Navigation
if 'active_section' not in st.session_state:
    st.session_state.active_section = None

# Navigation Buttons
if st.button("🔍 Blood Group Detection", key="bg_detection"):
    st.session_state.active_section = "blood_group_detection"
if st.button("🩸 Hematological Analysis", key="hema_analysis"):
    st.session_state.active_section = "hematological_analysis"

# Blood Group Detection Section
if st.session_state.active_section == "blood_group_detection":
    st.markdown("<h2 class='section-header'>Blood Group Detection</h2>", unsafe_allow_html=True)
    st.subheader("Choose Your Detection Method")
    method = st.radio("Select Method:", ["Fingerprint", "Blood Sample Image"], key="detection_method")

    if method == "Fingerprint":
        uploaded_file = st.file_uploader("Upload a Fingerprint Image for Blood Group Detection", type=["jpg", "png", "bmp"])
        if uploaded_file is not None:
            img = image.load_img(uploaded_file)
            st.image(img, caption="Uploaded Fingerprint Image", use_column_width=False, width=400)
            predicted_label = predict_blood_group(img, fingerprint_model)
            st.markdown(f"<div class='output'><b>Detected Blood Group: {blood_group_labels[predicted_label]}</b></div>", unsafe_allow_html=True)

    elif method == "Blood Sample Image":
        uploaded_file = st.file_uploader("Upload a Blood Sample Image for Blood Group Detection", type=["jpg", "png", "bmp"])
        if uploaded_file is not None:
            img = image.load_img(uploaded_file)
            st.image(img, caption="Uploaded Blood Sample Image", use_column_width=False, width=400)
            predicted_label = predict_blood_group(img, bloodsample_model)
            st.markdown(f"<div class='output'><b>Detected Blood Group: {blood_group_labels[predicted_label]}</b></div>", unsafe_allow_html=True)

# Hematological Analysis Section
if st.session_state.active_section == "hematological_analysis":
    st.markdown("<h2 class='section-header'>Hematological Analysis</h2>", unsafe_allow_html=True)
    st.subheader("Upload Blood Sample Image")
    uploaded_file = st.file_uploader("Upload Blood Sample Image for Hematological Analysis", type=["jpg", "png", "bmp"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Blood Sample Image for Hematological Analysis", use_column_width=False, width=400)
        
        # Perform analysis
        analysis_results = analyze_blood_sample(img)
        
        # Display results
        st.markdown("<div class='output'><b>Hematological Analysis Results:</b><br>", unsafe_allow_html=True)
        for key, value in analysis_results.items():
            st.write(f"**{key}:** {value}")
        st.markdown("</div>", unsafe_allow_html=True)
