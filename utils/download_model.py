import os
import tarfile
import urllib.request
from pathlib import Path

def download_and_extract_model(model_name, model_date, base_url, models_dir):
    model_tar = f"{model_name}.tar.gz"
    model_path = models_dir / model_name / "checkpoint"
    if not model_path.exists():
        print(f"Downloading {model_name}...")
        download_url = f"{base_url}/{model_date}/{model_tar}"
        tar_path = models_dir / model_tar
        urllib.request.urlretrieve(download_url, tar_path)
        with tarfile.open(tar_path) as tar:
            tar.extractall(path=models_dir)
        os.remove(tar_path)
        print(f"{model_name} downloaded and extracted.")

def download_label_map(label_url, save_path):
    if not save_path.exists():
        print("Downloading label map...")
        urllib.request.urlretrieve(label_url, save_path)
        print("Label map downloaded.")
