# 🧠 Streamlit Object Detection App

This is a real-time object detection app built with [Streamlit](https://streamlit.io/) and [TensorFlow 2](https://www.tensorflow.org/). It supports detection from images, videos, and live webcam input using the **SSD MobileNet V2 FPNLite 320x320** model from TensorFlow Model Zoo.

---

## 📦 Features

- 📷 **Live webcam detection**
- 🖼️ **Image file detection**
- 🎥 **Video file detection**
- ⚡️ Real-time performance using SSD MobileNet (lightweight + fast)
- 🧠 Powered by TensorFlow's pre-trained `saved_model`

---

## 🖼️ Demo

> Coming soon: [Streamlit Cloud Deployment Link](https://share.streamlit.io/...) 🔗

---

## 🔧 Project Structure

<pre>
    object-detection-app/
    │
    ├── app.py # Main Streamlit UI
    │
    ├── model_utils/ # Handlers for image, video, and live detection
    │ ├── image_model.py
    │ ├── video_model.py
    │ └── live_model.py
    │
    ├── utils/ # Shared helper functions
    │ ├── helpers.py
    │ └── download_model.py
    │
    ├── models/ # Saved models
    │ └── ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/
    │ ├── saved_model/
    │ └── mscoco_label_map.pbtxt
    │
    ├── test_media/ # Sample media files for testing
    │ ├── test_image.jpg
    │ └── test_video.mp4
    │
    ├── requirements.txt # Project dependencies
    └── README.md
</pre>


---

## 🚀 Installation & Running Locally

### 1. Clone the repository

```bash
git clone https://github.com/your-username/streamlit-object-detection-app.git
cd streamlit-object-detection-app
2. (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
3. Install dependencies
pip install -r requirements.txt
4. Run the Streamlit app
streamlit run app.py
🧠 Model Info
Model used: ssd_mobilenet_v2_fpnlite_320x320

Speed: Fast (good for real-time)

Input size: 320x320

Classes: COCO dataset (90 classes)

Accuracy: Medium (tradeoff for speed)

🧩 Dependencies
tensorflow

opencv-python

numpy

streamlit

(See requirements.txt for exact versions)

☁️ Deployment
To deploy to Streamlit Cloud:

Push your project to GitHub (include saved_model).

Go to https://streamlit.io/cloud

Link your GitHub repo and hit Deploy.

Make sure requirements.txt and app.py are in the root of your repo.

🙌 Acknowledgments
TensorFlow Model Zoo

Streamlit Community

