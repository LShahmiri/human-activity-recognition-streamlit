
# Human Activity Recognition (OpenPose + 3D CNN)


<img width="386" height="424" alt="image" src="https://github.com/user-attachments/assets/f6d3934c-0fac-46a8-8064-c0ca6424aeaf" />


This project performs human activity recognition using skeleton keypoints extracted from videos with **OpenPose (COCO-18)** and a **3D ResNet CNN** trained on the KTH action dataset.  (https://www.csc.kth.se/cvap/actions/)
The web interface is built using **Streamlit** and allows users to upload a video, extract pose keypoints, and obtain the predicted action label with confidence.

---

## 🔧 Features
- Upload a video file (`.mp4`, `.avi`, `.mov`)
- Automatic conversion → preview compatible with Streamlit
- Keypoint extraction using OpenPose COCO model
- JSON export of keypoints (optional)
- Action classification using a pretrained 3D CNN model
- Clean, styled Streamlit UI for desktop use

