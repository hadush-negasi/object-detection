import streamlit as st
import cv2
import tensorflow as tf
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

from utils.model_loader import load_model_and_labels
from utils.drawing import draw_boxes

#model_path = "models/ssd_mobilenet_v2_320x320_coco17_tpu-8"
#model, category_index = load_model_and_labels(model_path)

# Desired output width and height
OUTPUT_WIDTH = 640
OUTPUT_HEIGHT = 480

class ObjectDetectionTransformer(VideoTransformerBase):
    def transform(self, frame, model, category_index):
        img = frame.to_ndarray(format="bgr24")

        # Resize input to desired size for speed & output consistency
        img_resized = cv2.resize(img, (OUTPUT_WIDTH, OUTPUT_HEIGHT))

        input_tensor = tf.convert_to_tensor(np.expand_dims(img_resized, 0), dtype=tf.uint8)
        detections = model(input_tensor)

        img_resized = draw_boxes(img_resized, detections, category_index)

        return img_resized

def run():
    st.header("📷 Live Webcam Detection")

    #webrtc_streamer(
    #    key="object-detection",
    #    video_transformer_factory=ObjectDetectionTransformer,
    #    media_stream_constraints={
    #        "video": {"width": OUTPUT_WIDTH, "height": OUTPUT_HEIGHT},
    #        "audio": False  # disable audio capture
    #    },
    #    rtc_configuration={
    #        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    #    },
    #    async_processing=True,  # optional: can improve performance
    #    # optionally you can add desired video_frame_callback_fps to limit FPS
    #)
