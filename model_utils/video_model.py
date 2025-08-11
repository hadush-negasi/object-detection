import streamlit as st
import cv2
import tempfile
import tensorflow as tf
import numpy as np
from utils.model_loader import model, category_index
from utils.drawing import draw_boxes

#model_path = "models/ssd_mobilenet_v2_320x320_coco17_tpu-8"
#model, category_index = load_model_and_labels(model_path)

def run():
    st.subheader("🎥 Object Detection on Video")

    uploaded_file = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        cap = cv2.VideoCapture(tfile.name)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        output_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
        out = None

        stframe = st.empty()

        progress_bar = st.progress(0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        current_frame = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if out is None:
                h, w = frame.shape[:2]
                out = cv2.VideoWriter(output_path, fourcc, 20.0, (w, h))

            input_tensor = tf.convert_to_tensor(np.expand_dims(frame, 0), dtype=tf.uint8)
            detections = model(input_tensor)
            frame = draw_boxes(frame, detections, category_index)
        
            out.write(frame)
            stframe.image(frame, channels="BGR", use_column_width=True)
        
            current_frame += 1
            progress_bar.progress(min(current_frame / frame_count, 1.0))
        cap.release()
        out.release()

        with open(output_path, "rb") as f:
            st.download_button(
                label="📥 Download Processed Video",
                data=f,
                file_name="detected_video.mp4",
                mime="video/mp4"
            )
