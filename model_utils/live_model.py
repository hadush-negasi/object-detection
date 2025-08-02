import streamlit as st
import cv2
import tensorflow as tf
import numpy as np
from pathlib import Path
from utils.model_loader import load_model_and_labels
from utils.drawing import draw_boxes

model, category_index = load_model_and_labels()

def run():
    st.header("📷 Live Webcam Detection")

    run_webcam = st.checkbox("Start Webcam")

    if run_webcam:
        cap = cv2.VideoCapture(0)
        stframe = st.empty()

        while run_webcam and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            input_tensor = tf.convert_to_tensor(np.expand_dims(frame, 0), dtype=tf.uint8)
            detections = model(input_tensor)
            frame = draw_boxes(frame, detections, category_index)

            stframe.image(frame, channels="BGR", use_column_width=True)

        cap.release()
