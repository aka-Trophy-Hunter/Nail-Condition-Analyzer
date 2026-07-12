import os
import subprocess
import shutil
import sys
import tempfile
import requests
from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet152
from PIL import Image

st.image("11.jpg", use_column_width=True)

REPO_ROOT = Path(__file__).parent
YOLO_DIR = REPO_ROOT / "yolov5"
WEIGHTS_PATH = REPO_ROOT / "models" / "best.pt"

CLASSIFIER_URL = "https://huggingface.co/moneebaa/nailscan-resnet152/resolve/main/resnet152_nail.pt"
CLASSIFIER_PATH = REPO_ROOT / "models" / "resnet152_nail.pt"
CLASS_NAMES = ["Acral Lentiginous Melanoma", "Healthy Nail", "Onychogryphosis", "Blue Finger", "Clubbing", "Pitting"]

CLASSIFY_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@st.cache_resource
def load_classifier():
    if not CLASSIFIER_PATH.exists():
        CLASSIFIER_PATH.parent.mkdir(parents=True, exist_ok=True)
        with requests.get(CLASSIFIER_URL, stream=True) as r:
            r.raise_for_status()
            with open(CLASSIFIER_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

    model = resnet152(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(CLASSIFIER_PATH, map_location="cpu"))
    model.eval()
    return model


def classify_nail(image: Image.Image, model):
    tensor = CLASSIFY_TRANSFORM(image).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)[0]
    top_idx = int(torch.argmax(probs))
    return CLASS_NAMES[top_idx], float(probs[top_idx]) * 100


@st.cache_resource
def setup_yolov5():
    """Clone the yolov5 repo once (contains segment/predict.py we need)."""
    if not YOLO_DIR.exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/ultralytics/yolov5.git", str(YOLO_DIR)],
            check=True,
        )
    return True


def run_segmentation(image_path: str, out_dir: str):
    script = YOLO_DIR / "segment" / "predict.py"
    result = subprocess.run(
        [
            sys.executable, str(script),
            "--weights", str(WEIGHTS_PATH),
            "--source", image_path,
            "--project", out_dir,
            "--name", "run",
            "--exist-ok",
            "--conf-thres", "0.85",
            "--save-txt",
        ],
        cwd=str(YOLO_DIR),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-3000:] or result.stdout[-3000:])

    result_dir = Path(out_dir) / "run"
    image_result = None
    for f in result_dir.glob("*"):
        if f.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            image_result = f

    # A detection only counts if YOLO actually wrote a labels file with content.
    labels_dir = result_dir / "labels"
    n_detections = 0
    if labels_dir.exists():
        for txt_file in labels_dir.glob("*.txt"):
            with open(txt_file) as f:
                n_detections += len(f.readlines())

    if n_detections == 0:
        return None
    return image_result


st.title("NailScan")
st.caption("Stage 1: nail detection & segmentation. Disease classification is coming soon.")

with st.spinner("Setting up models..."):
    setup_yolov5()
    classifier_model = load_classifier()

file = st.file_uploader("Upload a photo of your nail(s)", type=["jpg", "jpeg", "png"])
st.caption("Works best with real, close-up photos of a nail. Not designed for illustrations, cartoons, or unrelated images.")

if file is None:
    st.info("Please upload an image to begin.")
else:
    image = Image.open(file).convert("RGB")
    st.image(image, caption="Uploaded image", use_column_width=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input.jpg")
        image.save(input_path)

        with st.spinner("Detecting and segmenting nail region..."):
            try:
                result_path = run_segmentation(input_path, tmpdir + "/out")
            except RuntimeError as e:
                st.error("Segmentation failed. Full error below:")
                st.code(str(e))
                result_path = None

        if result_path:
            st.image(str(result_path), caption="Detected nail region", use_column_width=True)
            st.success("Nail region detected successfully.")

            with st.spinner("Analyzing nail condition..."):
                predicted_class, confidence = classify_nail(image, classifier_model)

            if confidence < 40:
                st.warning(
                    f"Best guess: {predicted_class} ({confidence:.1f}% confidence) — "
                    "this is too low-confidence to be reliable. Try a clearer, closer photo of just the nail."
                )
            else:
                st.subheader(f"Prediction: {predicted_class}")
                st.write(f"Confidence: {confidence:.1f}%")
                st.caption(
                    "This is an automated screening tool trained on a public dataset, not a medical diagnosis. "
                    "Please consult a dermatologist for confirmation."
                )
        else:
            st.warning("No nail detected in this image. Try a clearer, closer photo of just the nail.")
