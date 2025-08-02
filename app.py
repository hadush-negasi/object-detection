import streamlit as st
from model_utils import image_model, video_model, live_model

st.title("Object Detection App")

option = st.sidebar.selectbox("Choose Detection Mode", ["Image", "Video", "Live Webcam"])

if option == "Image":
    image_model.run()
elif option == "Video":
    video_model.run()
elif option == "Live Webcam":
    live_model.run()
