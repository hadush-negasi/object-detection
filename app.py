import streamlit as st
st.set_page_config(
    page_title="Object Detection App",
    page_icon="🖼️",
    layout="wide"
)
from streamlit_option_menu import option_menu
from model_utils import image_model, video_model, live_model
import about

# Custom CSS to hide the Streamlit menu bar
hide_menu_style = """
    <style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .block-container {padding-top: 0px !important;}
    </style>
"""
# Inject the CSS into the app to hide top bar from streamlit
st.markdown(hide_menu_style, unsafe_allow_html=True)

# --- Navigation Menu ---
selected = option_menu(
    menu_title=None,  # No title, just the menu
    options=["Home / Image", "Video", "Live Webcam", "About"],
    icons=["image", "camera-reels", "camera-video", "info-circle"],  # Bootstrap icons
    menu_icon="cast",  # Optional icon for menu
    default_index=0,
    orientation="horizontal",  # Horizontal layout
    styles={
        "container": {"padding": "0!important", "background-color": "#f8f9fa"},
        "icon": {"color": "black", "font-size": "18px"},
        "nav-link": {
            "font-size": "16px",
            "text-align": "center",
            "margin": "0px",
            "--hover-color": "#eee",
        },
        "nav-link-selected": {"background-color": "#02ab21", "color": "white"},
    },
)

st.write("---")

# --- Page Rendering ---
if selected == "Home / Image":
    image_model.run()

elif selected == "Video":
    video_model.run()

elif selected == "Live Webcam":
    live_model.run()

elif selected == "About":
    about.run()
