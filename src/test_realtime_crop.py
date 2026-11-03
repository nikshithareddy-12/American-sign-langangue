import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import os


MODEL_PATH = "models/asl_clean_model.keras"
CLASSES_PATH = "clean_classes.npy"
MEDIAPIPE_MODEL = "hand_landmarker.task"

IMAGE_SIZE = 128
PADDING = 40


print("Loading model...")

model = tf.keras.models.load_model(MODEL_PATH)
classes = np.load(CLASSES_PATH, allow_pickle=True)

print("Model loaded.")
print("Classes:", classes)


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


camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("ERROR: Camera could not be opened.")
    exit()

print("\nCamera started.")
print("Show your hand.")
print("Press S to save the crop.")
print("Press Q or ESC to quit.")

frame_timestamp = 0
saved_number = 0


with HandLandmarker.create_from_options(options) as landmarker:

    while True:

        success, frame = camera.read()

        if not success:
            print("Could not read camera.")
            break

        model_frame = frame.copy()

        height, width, _ = frame.shape

        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        frame_timestamp += 33

        result = landmarker.detect_for_video(
            mp_image,
            frame_timestamp
        )

        prediction_text = "NO HAND DETECTED"
        confidence_text = ""

        if result.hand_landmarks:

            landmarks = result.hand_landmarks[0]

            x_coordinates = [
                int(landmark.x * width)
                for landmark in landmarks
            ]

            y_coordinates = [
                int(landmark.y * height)
                for landmark in landmarks
            ]

            x_min = max(0, min(x_coordinates) - PADDING)
            y_min = max(0, min(y_coordinates) - PADDING)

            x_max = min(width, max(x_coordinates) + PADDING)
            y_max = min(height, max(y_coordinates) + PADDING)

            # Draw box
            cv2.rectangle(
                frame,
                (x_min, y_min),
                (x_max, y_max),
                (0, 255, 0),
                2
            )


            hand_crop = model_frame[y_min:y_max, x_min:x_max]

            if hand_crop.size > 0:

                hand_crop = cv2.resize(
                    hand_crop,
                    (IMAGE_SIZE, IMAGE_SIZE)
                )

                hand_crop_rgb = cv2.cvtColor(
                    hand_crop,
                    cv2.COLOR_BGR2RGB
                )

                input_image = hand_crop_rgb.astype(
                    np.float32
                ) / 255.0

                # Add batch dimension
                input_image = np.expand_dims(
                    input_image,
                    axis=0
                )


                predictions = model.predict(
                    input_image,
                    verbose=0
                )[0]

                predicted_index = np.argmax(predictions)
                confidence = predictions[predicted_index]

                predicted_class = classes[predicted_index]

                prediction_text = str(predicted_class)
                confidence_text = f"{confidence * 100:.2f}%"


                cv2.putText(
                    frame,
                    f"Prediction: {predicted_class}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"Confidence: {confidence * 100:.2f}%",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )


                cv2.imshow(
                    "EXACT IMAGE SENT TO MODEL",
                    hand_crop
                )

        else:

            cv2.putText(
                frame,
                "NO HAND DETECTED",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

        # Show webcam
        cv2.imshow(
            "Realtime ASL Test",
            frame
        )


        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):

            if result.hand_landmarks and hand_crop.size > 0:

                saved_number += 1

                filename = f"realtime_crop_{saved_number}.jpg"

                cv2.imwrite(
                    filename,
                    hand_crop
                )

                print(
                    f"Saved: {filename}"
                )

        if key == ord("q") or key == 27:
            break

camera.release()
cv2.destroyAllWindows()

print("Camera closed.")