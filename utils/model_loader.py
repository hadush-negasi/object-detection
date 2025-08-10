import streamlit as st
import tensorflow as tf
from pathlib import Path
from object_detection.utils import label_map_util

# Cache the TensorFlow model (st.cache_resource)
@st.cache_resource
def load_model(model_path):
    model_dir = Path(model_path)
    return tf.saved_model.load(str(model_dir / "saved_model"))

# Cache the label map (st.cache_data)
@st.cache_data
def load_labels(label_path):
    return label_map_util.create_category_index_from_labelmap(
        str(label_path), use_display_name=True)

# Combined loader (uses cached sub-functions)
def load_model_and_labels(model_path):
    model = load_model(model_path)
    label_path = Path(model_path) / "mscoco_label_map.pbtxt"
    category_index = load_labels(label_path)
    return model, category_index