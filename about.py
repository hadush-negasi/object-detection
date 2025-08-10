import streamlit as st

def run():
    st.title("ℹ️ About This App")
    st.write("---")
    st.markdown("""
    ### 👨‍💻 Developer Info
    - **Name:** Hadush Negasi
    - **Portfolio:** [hadushnegasi.netlify.app/](https://hadushnegasi.netlify.app/)
    - **GitHub:** [github.com/hadush-negasi](https://github.com/hadush-negasi)
    - **LinkedIn:** [linkedin.com/in/hadush-brhane](https://www.linkedin.com/in/hadush-brhane/)
    
    ### 📜 About the Project
    This app demonstrates **real-time object detection** on:
    - Images
    - Videos
    - Live Webcam feed
    
    It uses **TensorFlow Object Detection API** with models such as:
    - `EfficientDet D1` for high accuracy
    - `SSD MobileNet v2` for fast inference
    
    ### 🚀 Features
    - Modern UI with icons
    - Download processed results
    - Supports multiple detection modes
    - Works directly in the browser
    """)