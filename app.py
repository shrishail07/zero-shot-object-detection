import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLOE
import supervision as sv

# Ensure OpenCV is properly imported
try:
    import cv2
except ImportError:
    st.error("OpenCV (cv2) is not installed. Please install it using 'pip install opencv-python-headless'.")

# Load YOLOE model
@st.cache_resource
def load_model():
    model = YOLOE("yoloe-v8l-seg.pt").cuda()  # Use GPU if available
    return model

model = load_model()

# Streamlit UI
st.title("YOLOE Object Detection Dashboard 🖼️🚀")
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
