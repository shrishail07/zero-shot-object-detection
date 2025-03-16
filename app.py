import streamlit as st
import numpy as np
import supervision as sv
from PIL import Image

# Ensure OpenCV is properly imported
try:
    import cv2
except ImportError:
    st.error("OpenCV (cv2) is not installed. Run 'pip install opencv-python-headless'.")
    st.stop()

# Ensure Ultralytics is properly imported
try:
    from ultralytics import YOLO
except ImportError:
    st.error("Ultralytics is not installed correctly. Run 'pip install ultralytics'.")
    st.stop()

# Load YOLOE model from model_loader.py
from model_loader import load_model
model = load_model()

# Streamlit UI
st.title("YOLOE Object Detection Dashboard 🚀")
st.write("Upload an image to detect objects using YOLOE-V8L-SEG.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Perform object detection
    st.write("Detecting objects...")
    results = model.predict(image)

    # Convert results to supervision detections
    detections = sv.Detections.from_ultralytics(results[0])

    # Annotate image
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    annotated_image = image.copy()
    annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

    # Convert PIL image to NumPy format for Streamlit
    annotated_image = np.array(annotated_image)

    # Display results
    st.image(annotated_image, caption="Detected Objects", use_column_width=True)

st.write("Built with ❤️ using Streamlit and YOLOE.")
