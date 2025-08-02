import streamlit as st
import cv2
import tempfile
import tensorflow as tf
import numpy as np
from pathlib import Path
from utils.model_loader import load_model_and_labels
from utils.drawing import draw_boxes

model, category_index = load_model_and_labels()
#from utils.helpers import load_model, load_labels, draw_boxes

def run():
    st.header("🎥 Object Detection on Video")

    uploaded_file = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        cap = cv2.VideoCapture(tfile.name)

        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            input_tensor = tf.convert_to_tensor(np.expand_dims(frame, 0), dtype=tf.uint8)
            detections = model(input_tensor)
            frame = draw_boxes(frame, detections, category_index)

            stframe.image(frame, channels="BGR", use_column_width=True)
        cap.release()
