import streamlit as st
import tensorflow as tf
from pathlib import Path
from object_detection.utils import label_map_util

@st.cache_resource
def load_model_and_labels():
    model_dir = Path("models/ssd_mobilenet_v2_320x320_coco17_tpu-8")
    model = tf.saved_model.load(str(model_dir / "saved_model"))
    label_path = model_dir / "mscoco_label_map.pbtxt"
    category_index = label_map_util.create_category_index_from_labelmap(
        str(label_path), use_display_name=True)
    return model, category_index
