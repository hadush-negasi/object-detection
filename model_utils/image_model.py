import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from utils.model_loader import model, category_index
from utils.drawing import draw_boxes

# Load the model once
#model_path = "models/ssd_mobilenet_v2_320x320_coco17_tpu-8"
#model, category_index = load_model_and_labels(model_path)

def run():
    st.subheader("🔍 Object Detection on Image")

    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image_np = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        with st.spinner("Processing image..."):
            input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.uint8)
            detections = model(input_tensor)

            image_np = draw_boxes(image_np, detections, category_index)

        st.image(image_np, channels="BGR", caption="Detected Image", use_column_width=True)

        # Convert to PNG for download
        _, buffer = cv2.imencode(".png", image_np)
        st.download_button(
            label="📥 Download Result",
            data=buffer.tobytes(),
            file_name="detected_image.png",
            mime="image/png"
        )
