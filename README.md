# American Sign Language (ASL) Detection System

An end-to-end computer vision and deep learning system that recognizes American Sign Language (ASL) hand signs in real time from a webcam stream, displaying the predicted letter or number alongside its confidence score.

---

## 🌟 Key Features

- **Real-Time Video Recognition:** Processes live camera feeds using OpenCV DirectShow (`cv2.CAP_DSHOW`) at $640 \times 480$ resolution.
- **MediaPipe ROI Isolation:** Employs Google's `MediaPipe HandLandmarker` to locate 21 3D hand keypoints and crop a centered, padded $128 \times 128$ hand region.
- **41 Supported Classes:** Classifies 15 numbers (`1` to `15`) and 26 letters (`A` to `Z`).
- **High Offline Accuracy:** Deep 2D Convolutional Neural Network (CNN) achieves **98.29% offline test set accuracy** (test loss: `0.0803`).
- **State-Machine Confirmation:** Locks confirmed predictions reaching $\ge 70\%$ confidence until the user releases the screen lock by pressing `SPACE`.

---

## 🖐️ Supported Gesture Classes (41 Total)

- **Numbers (15):** `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9`, `10`, `11`, `12`, `13`, `14`, `15`
- **Alphabets (26):** `A`, `B`, `C`, `D`, `E`, `F`, `G`, `H`, `I`, `J`, `K`, `L`, `M`, `N`, `O`, `P`, `Q`, `R`, `S`, `T`, `U`, `V`, `W`, `X`, `Y`, `Z`

---

## 🛠️ Technology Stack

- **Language:** Python 3.10 / 3.11 (Virtual Environment `C:\asl_env`)
- **Computer Vision:** OpenCV (`cv2`), MediaPipe (`hand_landmarker.task`)
- **Deep Learning Framework:** TensorFlow 2.21.0 / Keras
- **Data Processing:** NumPy, Scikit-Learn

---

## 📂 Project Structure

```
American sign lang/
│
├── dataset_v2/                     # Raw dataset image collection
├── cleaned_dataset_v2/             # Preprocessed & normalized hand crop images
│
├── models/
│   └── asl_clean_model.keras       # Trained Keras 2D CNN model weights
│
├── src/
│   ├── collection_data.py          # Interactive webcam dataset collector
│   ├── clean_dataset.py            # MediaPipe-based ROI dataset cleaner
│   ├── prepare_clean_dataset.py    # 70/15/15 train/val/test array exporter
│   ├── train_clean_model.py        # 2D CNN model trainer with data augmentation
│   ├── test_clean_model.py         # Offline test accuracy evaluator
│   ├── realtime_prediction.py      # Main real-time prediction application
│   └── check_dataset_quality.py    # Quality audit inspection utility
│
├── clean_classes.npy               # Array of 41 class label strings
├── clean_X_train.npy               # Training feature matrix (6010, 128, 128, 3)
├── clean_X_validation.npy          # Validation feature matrix (1271, 128, 128, 3)
├── clean_X_test.npy                # Testing feature matrix (1328, 128, 128, 3)
├── clean_y_train.npy               # Training target vector (6010,)
├── clean_y_validation.npy          # Validation target vector (1271,)
├── clean_y_test.npy                # Testing target vector (1328,)
│
├── hand_landmarker.task            # MediaPipe HandLandmarker task asset
├── PROJECT_DOCUMENTATION.md        # Comprehensive technical project documentation
└── README.md                       # Project overview and quickstart guide
```

---

## 🔄 How It Works

```
┌──────────────┐      ┌────────────────────┐      ┌────────────────────┐
│ Webcam Stream│ ───► │ MediaPipe ROI Crop │ ───► │ Keras 2D CNN Model │
│ (CAP_DSHOW)  │      │ (128x128 Normalize)│      │  (41 Classes)      │
└──────────────┘      └────────────────────┘      └─────────┬──────────┘
                                                            │
┌──────────────┐      ┌────────────────────┐                │
│  User SPACE  │ ◄─── │  State Machine     │ ◄──────────────┘
│ Freeze Lock  │      │ (70% Confidence)   │
└──────────────┘      └────────────────────┘
```

1. **Capture:** Video frame captured via `cv2.VideoCapture(0, cv2.CAP_DSHOW)` at $640 \times 480$.
2. **Hand Detection:** MediaPipe extracts 21 hand landmarks, computes a square bounding box with $40\text{px}$ padding.
3. **Preprocessing:** Image crop resized to $128 \times 128$, converted to RGB, and normalized ($/ 255.0$).
4. **Classification:** 2D CNN computes class probabilities.
5. **Confirmation:** If confidence $\ge 70\%$, prediction displays **`CONFIRMED`** and locks the screen until `SPACE` is pressed.

---

## 📊 Dataset & Training Summary

- **Total Processed Images:** 8,609 images
- **Train / Val / Test Split:** 70% / 15% / 15% (Stratified)
  - **Training Set:** 6,010 images
  - **Validation Set:** 1,271 images
  - **Test Set:** 1,328 images
- **Offline Model Performance:**
  - **Test Loss:** `0.0803`
  - **Test Accuracy:** `98.29%`

*Note: Offline test set accuracy (98.29%) measures static dataset performance. Realtime performance depends on live lighting, distance, and hand stability.*

---

## ⚡ Quick Start & Installation

### 1. Activate Virtual Environment
```powershell
C:\asl_env\Scripts\activate
```

### 2. Navigate to Project Directory
```powershell
cd "C:\Users\niksh\OneDrive\Desktop\American sign lang"
```

### 3. Run Real-Time Recognition Application
```powershell
python src\realtime_prediction.py
```

### 4. Run Offline Test Set Evaluation
```powershell
python src\test_clean_model.py
```

---

## 🎮 User Controls

| Key | Function |
| :--- | :--- |
| **Show Hand Sign** | Real-time classification actively predicts gesture. |
| **$\ge 70\%$ Confidence** | System freezes predicted class and displays `CONFIRMED`. |
| **`SPACE`** | Releases prediction freeze and resumes live scanning. |
| **`Q` / `ESC`** | Closes camera stream and exits application. |

---

## 🔧 Troubleshooting

- **Webcam Frame Issue (`MSMF can't grab frame`):** Resolved by using DirectShow (`cv2.CAP_DSHOW`).
- **MediaPipe Warning (`NORM_RECT`):** Harmless internal framework warning.
- **Missing Condition Subfolder (`dark: folder not found`):** Handled gracefully during dataset cleaning.

---

## 🔮 Limitations & Future Enhancements

### Current Limitations
- Supports single-hand gestures (`num_hands = 1`).
- Limited to the 41 trained classes (does not include numbers 16–20).
- Extreme low-light conditions may degrade MediaPipe hand detection.

### Future Improvements
- Expand dataset to include numbers 16–20 and complex word gestures.
- Add Text-to-Speech audio output (`pyttsx3`).
- Implement sentence assembly with backspace/clear capabilities.
- Export model to TensorFlow Lite for mobile deployment.

---

## 📄 Documentation

For full mathematical descriptions, architecture diagrams, and script breakdowns, refer to:
👉 [`PROJECT_DOCUMENTATION.md`](file:///C:/Users/niksh/OneDrive/Desktop/American%20sign%20lang/PROJECT_DOCUMENTATION.md)

---

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.
