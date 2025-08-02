import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from pathlib import Path
from utils.model_loader import load_model_and_labels
from utils.drawing import draw_boxes

model, category_index = load_model_and_labels()
#from utils.helpers import load_model, load_labels, draw_boxes

def run():
    st.header("🔍 Object Detection on Image")

    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image_np = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Prepare input
        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.uint8)
        detections = model(input_tensor)

        # Draw results
        image_np = draw_boxes(image_np, detections, category_index)

        st.image(image_np, channels="BGR", caption="Detected Image")
