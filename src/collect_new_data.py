import cv2
import os
import time

DATASET_DIR = "dataset_v2"

NUM_IMAGES = 30

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

BOX_X1 = 150
BOX_Y1 = 70
BOX_X2 = 490
BOX_Y2 = 450


camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

camera.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT)

if not camera.isOpened():
    print("ERROR: Camera could not be opened.")
    exit()



sign = input("Enter sign to collect: ").strip().upper()

if not sign:
    print("Invalid sign.")
    camera.release()
    exit()


folder = os.path.join(DATASET_DIR, sign, "normal")
os.makedirs(folder, exist_ok=True)


print()
print(f"Collecting NEW data for: {sign}")
print(f"Images: {NUM_IMAGES}")
print()
print("Keep your complete hand inside the guide box.")
print("Change position/angle slightly between captures.")
print("Press SPACE to capture.")
print("Press Q to quit.")

for i in range(3, 0, -1):

    ret, frame = camera.read()

    if not ret:
        continue

    cv2.putText(
        frame,
        f"Starting in {i}",
        (190, 240),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (0, 255, 0),
        3
    )

    cv2.imshow("ASL New Data Collection", frame)
    cv2.waitKey(1000)



count = 0

while count < NUM_IMAGES:

    ret, frame = camera.read()

    if not ret:
        continue

    display = frame.copy()

    # Guide rectangle
    cv2.rectangle(
        display,
        (BOX_X1, BOX_Y1),
        (BOX_X2, BOX_Y2),
        (0, 255, 0),
        2
    )

    cv2.putText(
        display,
        f"Sign: {sign}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

    cv2.putText(
        display,
        f"Images: {count}/{NUM_IMAGES}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        display,
        "Keep hand inside box",
        (170, 465),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        display,
        "SPACE = capture | Q = quit",
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow("ASL New Data Collection", display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    if key == 32:

        filename = os.path.join(
            folder,
            f"{sign}_new_{count + 1:03d}.jpg"
        )

        cv2.imwrite(filename, frame)

        count += 1

        print(f"Saved {count}/{NUM_IMAGES}")

        time.sleep(0.12)



camera.release()
cv2.destroyAllWindows()

print()
print(f"Finished {sign}: {count} images")
print(f"Saved in: {folder}")