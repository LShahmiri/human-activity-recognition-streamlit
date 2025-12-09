
# Human Activity Recognition (OpenPose + 3D CNN)


 <img width="477" height="1232" alt="image" src="https://github.com/user-attachments/assets/64b33bef-c796-4d21-9cd3-014a6e4fa551" />


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

