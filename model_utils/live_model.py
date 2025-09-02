import os
import streamlit as st
import cv2
import tensorflow as tf
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from twilio.rest import Client

from utils.model_loader import get_model
from utils.drawing import draw_boxes

model, category_index = get_model("live") # SSD Mobilenet model

# Desired output width and height
OUTPUT_WIDTH = 640
OUTPUT_HEIGHT = 480


# ---- TWILIO TURN SERVER FETCHER ----
@st.cache_resource  # cache to avoid hitting Twilio API too often
def get_twilio_ice_servers():
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    if not account_sid or not auth_token:
        st.error("❌ Twilio credentials not found. Please set Hugging Face secrets.")
        return [{"urls": ["stun:stun.l.google.com:19302"]}]  # fallback to STUN only

    client = Client(account_sid, auth_token)
    token = client.tokens.create()
    return token.ice_servers


# ---- VIDEO TRANSFORMER ----
class ObjectDetectionTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = model
        self.category_index = category_index
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Resize input to desired size for speed & output consistency
        img_resized = cv2.resize(img, (OUTPUT_WIDTH, OUTPUT_HEIGHT))

        input_tensor = tf.convert_to_tensor(np.expand_dims(img_resized, 0), dtype=tf.uint8)
        detections = model(input_tensor)

        img_resized = draw_boxes(img_resized, detections, category_index)

        return img_resized


# ---- MAIN RUN ----
def run():
    st.header("📷 Live Webcam Detection")

    ice_servers = get_twilio_ice_servers()

    webrtc_streamer(
        key="object-detection",
        video_transformer_factory=ObjectDetectionTransformer,
        media_stream_constraints={
            "video": {"width": OUTPUT_WIDTH, "height": OUTPUT_HEIGHT},
            "audio": False,
        },
        rtc_configuration={
            "iceServers": ice_servers,
            "iceTransportPolicy": "all",  # or "relay" to force TURN only
        },
        async_processing=True,
        video_html_attrs={
            "style": {
                "width": "70%",
                "max-width": "800px",
                "height": "auto",
                "display": "block",
                "margin": "0 auto",
            },
            "controls": False,
            "autoPlay": True,
        }
    )
