# ============================================================
# ASL REALTIME PREDICTION
# ============================================================

import os
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "models/asl_clean_model.keras"
CLASSES_PATH = "clean_classes.npy"
MEDIAPIPE_MODEL = "hand_landmarker.task"

IMAGE_SIZE = 128
PADDING = 40

CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# Confirmation threshold (70%)
CONFIDENCE_THRESHOLD = 0.70


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

if not os.path.exists(MODEL_PATH):
    print("ERROR: Model file not found:")
    print(MODEL_PATH)
    raise SystemExit

if not os.path.exists(CLASSES_PATH):
    print("ERROR: Classes file not found:")
    print(CLASSES_PATH)
    raise SystemExit

if not os.path.exists(MEDIAPIPE_MODEL):
    print("ERROR: MediaPipe model not found:")
    print(MEDIAPIPE_MODEL)
    raise SystemExit


# ============================================================
# LOAD MODEL & CLASSES
# ============================================================

print("============================================")
print("       ASL REALTIME PREDICTION SYSTEM       ")
print("============================================")
print()

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

classes = np.load(CLASSES_PATH, allow_pickle=True)

print("Model loaded successfully!")
print(f"Total Classes: {len(classes)}")
print("Classes list:")
print(list(classes))
print()


# ============================================================
# MEDIAPIPE HAND LANDMARKER SETUP
# ============================================================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MEDIAPIPE_MODEL),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.6
)

landmarker = HandLandmarker.create_from_options(options)
print("MediaPipe HandLandmarker ready!")
print()


# ============================================================
# INITIALIZE CAMERA
# ============================================================

print("Starting camera feed...")
camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

if not camera.isOpened():
    print("ERROR: Could not open camera with CAP_DSHOW.")
    landmarker.close()
    raise SystemExit

print("Camera active!")
print()
print("============================================")
print("CONTROLS & INSTRUCTIONS")
print("============================================")
print("Show your ASL sign to the camera.")
print(f"Threshold: {CONFIDENCE_THRESHOLD * 100:.0f}% confidence = CONFIRMED")
print("SPACE = Resume prediction after confirmation")
print("Q / ESC = Exit program")
print("============================================")
print()


# ============================================================
# STATE VARIABLES
# ============================================================

timestamp_ms = 0
confirmed = False
confirmed_class = ""
confirmed_confidence = 0.0


# ============================================================
# MAIN APPLICATION LOOP
# ============================================================

while True:
    # --------------------------------------------------------
    # READ CAMERA FRAME
    # --------------------------------------------------------
    success, frame = camera.read()
    if not success:
        print("ERROR: Could not read camera frame.")
        break

    # Flip horizontally for mirror display
    frame = cv2.flip(frame, 1)
    display_frame = frame.copy()
    frame_height, frame_width = frame.shape[:2]

    # ========================================================
    # STATE: CONFIRMED (FROZEN PREDICTION)
    # ========================================================
    if confirmed:
        # Display frozen confidence score
        cv2.putText(
            display_frame,
            f"Confidence: {confirmed_confidence * 100:.1f}%",
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        # Display confirmed class label
        cv2.putText(
            display_frame,
            f"{confirmed_class}  CONFIRMED",
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 0),
            3
        )

        # Display resume instruction
        cv2.putText(
            display_frame,
            "Press SPACE for next sign",
            (20, 155),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (200, 200, 200),
            2
        )

        # Show camera window with frozen overlay
        cv2.imshow("ASL Realtime Prediction", display_frame)

        # Handle keyboard input during CONFIRMED state
        key = cv2.waitKey(1) & 0xFF
        if key == 32:  # SPACE key releases confirmed freeze
            confirmed = False
            confirmed_class = ""
            confirmed_confidence = 0.0
        elif key == ord("q") or key == 27:  # Q or ESC key exits
            break

        continue

    # ========================================================
    # STATE: PREDICTING (ACTIVE INFERENCE)
    # ========================================================
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    timestamp_ms += 33

    # Detect hand landmarks using MediaPipe HandLandmarker
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    if result.hand_landmarks:
        hand_landmarks = result.hand_landmarks[0]

        # Extract normalized coordinates to pixel coordinates
        xs = [lm.x * frame_width for lm in hand_landmarks]
        ys = [lm.y * frame_height for lm in hand_landmarks]

        # Calculate bounding box with padding
        x1 = max(0, int(min(xs)) - PADDING)
        y1 = max(0, int(min(ys)) - PADDING)
        x2 = min(frame_width, int(max(xs)) + PADDING)
        y2 = min(frame_height, int(max(ys)) + PADDING)

        width = x2 - x1
        height = y2 - y1

        if width > 0 and height > 0:
            # Make bounding box square
            size = max(width, height)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            sx1 = cx - size // 2
            sy1 = cy - size // 2
            sx2 = sx1 + size
            sy2 = sy1 + size

            # Boundary adjustments
            if sx1 < 0:
                sx2 -= sx1
                sx1 = 0
            if sy1 < 0:
                sy2 -= sy1
                sy1 = 0
            if sx2 > frame_width:
                sx1 -= (sx2 - frame_width)
                sx2 = frame_width
            if sy2 > frame_height:
                sy1 -= (sy2 - frame_height)
                sy2 = frame_height

            sx1 = max(0, sx1)
            sy1 = max(0, sy1)
            sx2 = min(frame_width, sx2)
            sy2 = min(frame_height, sy2)

            crop = frame[sy1:sy2, sx1:sx2]

            if crop.size > 0:
                # Resize to model input shape (128 x 128)
                resized = cv2.resize(crop, (IMAGE_SIZE, IMAGE_SIZE))
                resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                input_image = resized_rgb.astype(np.float32) / 255.0
                input_image = np.expand_dims(input_image, axis=0)

                # Predict gesture probabilities
                predictions = model.predict(input_image, verbose=0)[0]
                predicted_index = int(np.argmax(predictions))
                confidence = float(predictions[predicted_index])
                predicted_class = str(classes[predicted_index])

                # Draw bounding box and hand landmarks on display frame
                cv2.rectangle(display_frame, (sx1, sy1), (sx2, sy2), (0, 255, 0), 2)
                for lm in hand_landmarks:
                    px = int(lm.x * frame_width)
                    py = int(lm.y * frame_height)
                    cv2.circle(display_frame, (px, py), 4, (0, 255, 0), -1)

                # Check if prediction meets or exceeds 70% threshold
                if confidence >= CONFIDENCE_THRESHOLD:
                    # Confirm prediction and transition to CONFIRMED state
                    confirmed = True
                    confirmed_class = predicted_class
                    confirmed_confidence = confidence

                    cv2.putText(
                        display_frame,
                        f"Confidence: {confirmed_confidence * 100:.1f}%",
                        (20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2
                    )
                    cv2.putText(
                        display_frame,
                        f"{confirmed_class}  CONFIRMED",
                        (20, 105),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5,
                        (0, 255, 0),
                        3
                    )
                    cv2.putText(
                        display_frame,
                        "Press SPACE for next sign",
                        (20, 155),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (200, 200, 200),
                        2
                    )
                else:
                    # Unconfirmed prediction (< 70%)
                    cv2.putText(
                        display_frame,
                        f"Confidence: {confidence * 100:.1f}%",
                        (20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2
                    )
                    cv2.putText(
                        display_frame,
                        predicted_class,
                        (20, 105),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5,
                        (0, 255, 255),
                        3
                    )
    else:
        # No hand detected in active predicting state
        cv2.putText(
            display_frame,
            "NO HAND DETECTED",
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            3
        )

    # Display video frame
    cv2.imshow("ASL Realtime Prediction", display_frame)

    # Handle keyboard input during PREDICTING state
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or key == 27:
        break


# ============================================================
# RESOURCE CLEANUP
# ============================================================

camera.release()
landmarker.close()
cv2.destroyAllWindows()

print()
print("============================================")
print("Program closed cleanly.")
print("============================================")