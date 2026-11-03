import os
import cv2
import csv
import shutil
import numpy as np
import mediapipe as mp


DATASET_DIR = "dataset"
TASK_FILE = "hand_landmarker.task"

REPORT_FILE = "dataset_quality_report.csv"
FLAGGED_DIR = "quality_flags"

BLUR_THRESHOLD = 80
MIN_HAND_AREA_RATIO = 0.05
MAX_HAND_AREA_RATIO = 0.85

# Number of most unusual poses to flag per class
POSE_OUTLIER_PERCENT = 10



BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=TASK_FILE),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

landmarker = HandLandmarker.create_from_options(options)



def calculate_blur(image):
    """
    Higher value = sharper image.
    Lower value = blurrier image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def calculate_brightness(image):
    """
    Average brightness from 0 to 255.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def get_hand_info(image):
    """
    Detect hand and return:
    - normalized landmark vector
    - hand bounding box
    - hand area ratio
    - whether hand touches image boundary
    """

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = landmarker.detect(mp_image)

    if not result.hand_landmarks:
        return None

    landmarks = result.hand_landmarks[0]

    h, w = image.shape[:2]

    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]

    x_min = max(0, int(min(xs) * w))
    x_max = min(w - 1, int(max(xs) * w))

    y_min = max(0, int(min(ys) * h))
    y_max = min(h - 1, int(max(ys) * h))

    bbox_width = x_max - x_min
    bbox_height = y_max - y_min

    bbox_area = bbox_width * bbox_height
    image_area = w * h

    area_ratio = bbox_area / image_area

   
    touches_edge = (
        x_min <= 2 or
        y_min <= 2 or
        x_max >= w - 3 or
        y_max >= h - 3
    )

    

    wrist_x = landmarks[0].x
    wrist_y = landmarks[0].y
    wrist_z = landmarks[0].z

    points = []

    for lm in landmarks:
        points.extend([
            lm.x - wrist_x,
            lm.y - wrist_y,
            lm.z - wrist_z
        ])

    vector = np.array(points, dtype=np.float32)

    # Normalize scale
    scale = np.linalg.norm(vector)

    if scale > 0:
        vector = vector / scale

    return {
        "landmark_vector": vector,
        "area_ratio": area_ratio,
        "touches_edge": touches_edge,
        "bbox": (x_min, y_min, x_max, y_max)
    }



records = []

print("\n==========================================")
print("       DATASET QUALITY CHECKER")
print("==========================================\n")

if not os.path.exists(DATASET_DIR):
    print(f"ERROR: Dataset folder not found: {DATASET_DIR}")
    landmarker.close()
    exit()

classes = sorted([
    folder for folder in os.listdir(DATASET_DIR)
    if os.path.isdir(os.path.join(DATASET_DIR, folder))
])

print(f"Classes found: {len(classes)}")
print(classes)
print()


for sign in classes:

    sign_path = os.path.join(DATASET_DIR, sign)

    for condition in ["normal", "dark"]:

        condition_path = os.path.join(sign_path, condition)

        if not os.path.exists(condition_path):
            continue

        image_files = [
            f for f in os.listdir(condition_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        print(f"Checking {sign}/{condition}: {len(image_files)} images")

        for filename in image_files:

            path = os.path.join(condition_path, filename)

            image = cv2.imread(path)

            if image is None:
                records.append({
                    "sign": sign,
                    "condition": condition,
                    "filename": filename,
                    "status": "BAD_IMAGE",
                    "reason": "Cannot read image",
                    "blur": 0,
                    "brightness": 0,
                    "hand_area": 0,
                    "edge": "unknown"
                })
                continue

            blur = calculate_blur(image)
            brightness = calculate_brightness(image)
            hand_info = get_hand_info(image)

            reasons = []

            if blur < BLUR_THRESHOLD:
                reasons.append("BLUR")

            if brightness < 35:
                reasons.append("VERY_DARK")

            if brightness > 225:
                reasons.append("VERY_BRIGHT")

            if hand_info is None:

                reasons.append("NO_HAND")

                records.append({
                    "sign": sign,
                    "condition": condition,
                    "filename": filename,
                    "status": "FLAG",
                    "reason": ";".join(reasons),
                    "blur": round(blur, 2),
                    "brightness": round(brightness, 2),
                    "hand_area": 0,
                    "edge": "unknown",
                    "pose_vector": None
                })

                continue

            area_ratio = hand_info["area_ratio"]
            touches_edge = hand_info["touches_edge"]

            if area_ratio < MIN_HAND_AREA_RATIO:
                reasons.append("HAND_TOO_SMALL")

            if area_ratio > MAX_HAND_AREA_RATIO:
                reasons.append("HAND_TOO_LARGE")

            if touches_edge:
                reasons.append("HAND_AT_EDGE")

            status = "FLAG" if reasons else "GOOD"

            records.append({
                "sign": sign,
                "condition": condition,
                "filename": filename,
                "status": status,
                "reason": ";".join(reasons),
                "blur": round(blur, 2),
                "brightness": round(brightness, 2),
                "hand_area": round(area_ratio, 4),
                "edge": touches_edge,
                "pose_vector": hand_info["landmark_vector"]
            })

print("\nChecking for unusual hand poses...\n")

for sign in classes:

    sign_records = [
        r for r in records
        if r["sign"] == sign
        and r["pose_vector"] is not None
    ]

    if len(sign_records) < 5:
        continue

    vectors = np.array([
        r["pose_vector"]
        for r in sign_records
    ])


    median_pose = np.median(vectors, axis=0)

    distances = []

    for record in sign_records:

        distance = np.linalg.norm(
            record["pose_vector"] - median_pose
        )

        distances.append((distance, record))

    distances.sort(
        key=lambda x: x[0],
        reverse=True
    )

    number_to_flag = max(
        1,
        int(len(distances) * POSE_OUTLIER_PERCENT / 100)
    )

    for distance, record in distances[:number_to_flag]:

        if "POSE_OUTLIER" not in record["reason"]:
            if record["reason"]:
                record["reason"] += ";POSE_OUTLIER"
            else:
                record["reason"] = "POSE_OUTLIER"

            record["status"] = "FLAG"

        record["pose_distance"] = round(float(distance), 4)



print("Creating flagged image folders...\n")

if os.path.exists(FLAGGED_DIR):
    shutil.rmtree(FLAGGED_DIR)

os.makedirs(FLAGGED_DIR, exist_ok=True)


flagged_count = 0
good_count = 0


for record in records:

    if record["status"] == "FLAG":

        source = os.path.join(
            DATASET_DIR,
            record["sign"],
            record["condition"],
            record["filename"]
        )

        reason_folder = record["reason"].replace(";", "_")

        destination_dir = os.path.join(
            FLAGGED_DIR,
            record["sign"],
            reason_folder
        )

        os.makedirs(destination_dir, exist_ok=True)

        destination = os.path.join(
            destination_dir,
            record["filename"]
        )

        shutil.copy2(source, destination)

        flagged_count += 1

    else:
        good_count += 1



fieldnames = [
    "sign",
    "condition",
    "filename",
    "status",
    "reason",
    "blur",
    "brightness",
    "hand_area",
    "edge",
    "pose_distance"
]

with open(
    REPORT_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()

    for record in records:

        writer.writerow({
            "sign": record["sign"],
            "condition": record["condition"],
            "filename": record["filename"],
            "status": record["status"],
            "reason": record["reason"],
            "blur": record.get("blur", ""),
            "brightness": record.get("brightness", ""),
            "hand_area": record.get("hand_area", ""),
            "edge": record.get("edge", ""),
            "pose_distance": record.get("pose_distance", "")
        })



print("\n==========================================")
print("             QUALITY SUMMARY")
print("==========================================")

print(f"Total images checked : {len(records)}")
print(f"Good images           : {good_count}")
print(f"Flagged images        : {flagged_count}")

if len(records) > 0:
    percentage = flagged_count / len(records) * 100
    print(f"Flagged percentage    : {percentage:.2f}%")

print("\nFiles created:")
print(f"  {REPORT_FILE}")
print(f"  {FLAGGED_DIR}/")

print("\nDone!")

landmarker.close()