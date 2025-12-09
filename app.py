import streamlit as st
import cv2
import os
import json
import tempfile
from zipfile import ZipFile
import pandas as pd
import subprocess
import uuid
import numpy as np
import zipfile
import tensorflow as tf

# ----------------------------------------------------------
# PROFESSIONAL DESKTOP UI DESIGN (CSS)
# ----------------------------------------------------------
st.markdown("""
<style>

html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
}

.block-container {
    padding-top: 1.5rem;
    padding-left: 3rem;
    padding-right: 3rem;
}

h1 {
    font-size: 2rem !important;
    font-weight: 700 !important;
}

.card {
    padding: 1.2rem;
    border-radius: 12px;
    background-color: #f8f9fc;
    border: 1px solid #e1e8f0;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.04);
    margin-top: 0.8rem;
}

.stButton>button {
    border-radius: 8px;
    padding: 0.6rem 1.2rem;
    background-color: #2c6bed;
    color: white;
    font-size: 1rem;
    border: none;
}

.stButton>button:hover {
    background-color: #1f54ba;
}

video {
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------
# Convert ONLY for display (not for JSON)
# ----------------------------------------------------------
def convert_for_display(uploaded_video):
    temp_input = f"/tmp/{uuid.uuid4().hex}.avi"
    with open(temp_input, "wb") as f:
        f.write(uploaded_video.getvalue())

    temp_output = f"/tmp/{uuid.uuid4().hex}.mp4"
    cmd = f"ffmpeg -y -i {temp_input} -vcodec libx264 -acodec aac {temp_output}"
    subprocess.run(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    return temp_output


# ----------------------------------------------------------
# Download COCO model on startup
# ----------------------------------------------------------
MODEL_DIR = "models/coco"
PROTOTXT = f"{MODEL_DIR}/pose_deploy_linevec.prototxt"
CAFFEMODEL = f"{MODEL_DIR}/pose_iter_440000.caffemodel"
os.makedirs(MODEL_DIR, exist_ok=True)

if not os.path.isfile(PROTOTXT):
    os.system(f"wget -O {PROTOTXT} https://raw.githubusercontent.com/nik1806/Human-Pose-Detection/master/model/coco/pose_deploy_linevec.prototxt")
if not os.path.isfile(CAFFEMODEL):
    os.system(f"wget -O {CAFFEMODEL} https://github.com/nik1806/Human-Pose-Detection/raw/master/model/coco/pose_iter_440000.caffemodel")

net = cv2.dnn.readNetFromCaffe(PROTOTXT, CAFFEMODEL)


# ----------------------------------------------------------
# COCO LABELS
# ----------------------------------------------------------
COCO_LABELS = [
    "nose", "neck",
    "right_shoulder", "right_elbow", "right_wrist",
    "left_shoulder", "left_elbow", "left_wrist",
    "right_hip", "right_knee", "right_ankle",
    "left_hip", "left_knee", "left_ankle",
    "right_eye", "left_eye",
    "right_ear", "left_ear"
]


# ----------------------------------------------------------
# UI HEADER
# ----------------------------------------------------------
st.title(" Human Activity Recognition (OpenPose + 3D CNN)")
st.markdown("### Upload a video to analyse the action and view prediction results.")


# ----------------------------------------------------------
# FILE UPLOADER
# ----------------------------------------------------------
uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])


# ----------------------------------------------------------
# LAYOUT: TWO COLUMN DESIGN
# ----------------------------------------------------------
if uploaded_video is not None:

    col1, col2 = st.columns([1.2, 1])

    # ------------------------------------------------------
    # LEFT COLUMN: VIDEO PREVIEW
    # ------------------------------------------------------
    with col1:
        st.subheader("📽 Video Preview")
        preview_path = convert_for_display(uploaded_video)
        st.video(preview_path)

    # ------------------------------------------------------
    # RIGHT COLUMN: JSON GENERATION
    # ------------------------------------------------------
    with col2:
        st.subheader("🧩 Pose Extraction (OpenPose COCO)")
        st.info("⏳ Extracting pose keypoints...")

        # save original video (not converted)
        temp_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".avi").name
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_video.getvalue())

        cap = cv2.VideoCapture(temp_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25

        output_json_dir = tempfile.mkdtemp()
        frame_id = 0
        skip_frames = 15

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_id % skip_frames != 0:
                frame_id += 1
                continue

            h, w = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 1/255.0, (224,224), (0,0,0), swapRB=False, crop=False)
            net.setInput(blob)
            output = net.forward()

            keypoints = []
            for i, label in enumerate(COCO_LABELS):
                probMap = output[0, i, :, :]
                _, prob, _, point = cv2.minMaxLoc(probMap)
                x = int(point[0] * w / output.shape[3])
                y = int(point[1] * h / output.shape[2])
                keypoints.append({
                    "id": i,
                    "name": label,
                    "x": x,
                    "y": y,
                    "confidence": float(prob)
                })

            results = {
                "frame": frame_id,
                "time": round(frame_id / fps, 3),
                "keypoints": keypoints
            }

            json_file = os.path.join(output_json_dir, f"frame_{frame_id:05d}.json")
            with open(json_file, "w") as f:
                json.dump(results, f, indent=2)

            frame_id += 1

        cap.release()

        # ZIP JSON FILES
        zip_path = "pose_keypoints.zip"
        with ZipFile(zip_path, 'w') as zipf:
            for file in os.listdir(output_json_dir):
                zipf.write(os.path.join(output_json_dir, file), arcname=file)

        st.success("✅ Pose JSON extracted!")

        with open(zip_path, "rb") as f:
            st.download_button("⬇️ Download Keypoints ZIP", f, "pose_keypoints.zip")


# ----------------------------------------------------------
# ACTION CLASSIFICATION
# ----------------------------------------------------------
st.header(" Action Classification (3D CNN)")


MODEL_PATH = "saved_model/best_model_fold_1.keras"

@st.cache_resource
def load_model_cached():
    return tf.keras.models.load_model(MODEL_PATH)

if os.path.exists(MODEL_PATH):
    model = load_model_cached()
else:
    model = None
    st.warning("Model file not found.")


# ----------------------------------------------------------
# Preprocess JSON for model
# ----------------------------------------------------------
def preprocess_json_sequence(json_files, T=30, conf_thresh=0.20):
    frames = []
    all_x, all_y = [], []

    json_files = sorted(json_files, key=lambda f: int(os.path.basename(f).split("_")[1].split(".")[0]))

    for fpath in json_files:
        with open(fpath, "r") as f:
            d = json.load(f)

        kp = []
        for p in d["keypoints"]:
            x, y, c = p["x"], p["y"], p["confidence"]
            kp.append([x, y, c])
            all_x.append(x)
            all_y.append(y)

        frames.append(kp)

    frames = np.array(frames)

    max_x = max(all_x) if max(all_x) != 0 else 1
    max_y = max(all_y) if max(all_y) != 0 else 1

    frames[:, :, 0] /= max_x
    frames[:, :, 1] /= max_y

    low_conf = frames[:, :, 2] < conf_thresh
    frames[low_conf] = 0

    if len(frames) < T:
        idx = np.linspace(0, len(frames)-1, T)
        frames = np.array([frames[int(i)] for i in idx])
    else:
        idx = np.linspace(0, len(frames)-1, T).astype(int)
        frames = frames[idx]

    return frames.reshape(T, 18, 3, 1)


CLASS_MAP = {
    0: "boxing",
    1: "handclapping",
    2: "handwaving",
    3: "jogging",
    4: "running",
    5: "walk"
}


# ----------------------------------------------------------
# RUN CLASSIFICATION
# ----------------------------------------------------------
if uploaded_video is not None and model is not None:

    extract_dir = tempfile.mkdtemp()
    with zipfile.ZipFile("pose_keypoints.zip", "r") as zip_ref:
        zip_ref.extractall(extract_dir)

    json_files = [
        os.path.join(extract_dir, f)
        for f in os.listdir(extract_dir)
        if f.endswith(".json")
    ]

    if len(json_files) > 0:

        sample = preprocess_json_sequence(json_files)
        sample_input = np.expand_dims(sample, axis=0)

        preds = model.predict(sample_input)[0]
        class_id = np.argmax(preds)
        class_name = CLASS_MAP[class_id]
        prob = preds[class_id]

        # RESULT CARD
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write(f"###  Predicted Action: **{class_name}**")
        st.write(f"####  Confidence: **{prob*100:.2f}%**")
        st.markdown('</div>', unsafe_allow_html=True)

      