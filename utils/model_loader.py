import streamlit as st
import tensorflow as tf
from pathlib import Path
from object_detection.utils import label_map_util

# Cache the TensorFlow model (st.cache_resource)
SSD_PATH = "src/models/ssd_mobilenet_v2_320x320_coco17_tpu-8"
FASTER_RCNN_PATH = "src/models/faster_rcnn_resnet101_v1_1024x1024_coco17_tpu-8"

def load_model(path):
    model_dir = Path(path)
    return tf.saved_model.load(str(model_dir / "saved_model"))

# Cache the label map (st.cache_data)

def load_labels(label_path):
    return label_map_util.create_category_index_from_labelmap(
        str(label_path), use_display_name=True)

# Combined loader (uses cached sub-functions)
@st.cache_resource
def load_model_and_labels(path):
    model = load_model(path)
    label_path = Path(path) / "mscoco_label_map.pbtxt"
    category_index = load_labels(label_path)
    return model, category_index

faster_rcnn_model, faster_rcnn_label = load_model_and_labels(FASTER_RCNN_PATH)
ssd_model, ssd_label = load_model_and_labels(SSD_PATH)

def get_model(task: str = "image"):
    if task == "image":
        return faster_rcnn_model, faster_rcnn_label
    else:
        return ssd_model, ssd_label
