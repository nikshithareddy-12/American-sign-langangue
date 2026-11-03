import os
import cv2
import mediapipe as mp


DATASET_FOLDER = "dataset_v2"
CLEANED_FOLDER = "cleaned_dataset_v2"

IMAGE_SIZE = 128

# Extra space around the hand
PADDING = 40



MODEL_FILE = "hand_landmarker.task"

if not os.path.exists(MODEL_FILE):
    print("ERROR: hand_landmarker.task was not found.")
    print("Make sure it is in the project folder.")
    exit()


from mediapipe.tasks import python
from mediapipe.tasks.python import vision


base_options = python.BaseOptions(
    model_asset_path=MODEL_FILE
)


options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6
)


landmarker = vision.HandLandmarker.create_from_options(
    options
)



os.makedirs(
    CLEANED_FOLDER,
    exist_ok=True
)



total_images = 0
cleaned_images = 0
rejected_images = 0


print()
print("============================================")
print("       CLEANING ASL DATASET")
print("============================================")
print()



for sign in sorted(os.listdir(DATASET_FOLDER)):

    sign_path = os.path.join(
        DATASET_FOLDER,
        sign
    )

    # Ignore files
    if not os.path.isdir(sign_path):
        continue


    print(f"\nProcessing sign: {sign}")


    for condition in ["normal", "dark"]:

        input_folder = os.path.join(
            sign_path,
            condition
        )

        output_folder = os.path.join(
            CLEANED_FOLDER,
            sign,
            condition
        )

        os.makedirs(
            output_folder,
            exist_ok=True
        )


        
        if not os.path.exists(input_folder):

            print(
                f"  {condition}: folder not found"
            )

            continue



        image_files = [
            f
            for f in os.listdir(input_folder)
            if f.lower().endswith(
                (".jpg", ".jpeg", ".png")
            )
        ]


        condition_cleaned = 0
        condition_rejected = 0


        for filename in image_files:

            total_images += 1


            input_path = os.path.join(
                input_folder,
                filename
            )



            image = cv2.imread(
                input_path
            )


            if image is None:

                rejected_images += 1
                condition_rejected += 1

                continue


            height, width, _ = image.shape



            rgb_image = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )



            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_image
            )


            result = landmarker.detect(
                mp_image
            )



            if not result.hand_landmarks:

                rejected_images += 1
                condition_rejected += 1

                continue



            hand = result.hand_landmarks[0]



            x_coordinates = []

            y_coordinates = []


            for landmark in hand:

                x = int(
                    landmark.x * width
                )

                y = int(
                    landmark.y * height
                )

                x = max(
                    0,
                    min(x, width - 1)
                )

                y = max(
                    0,
                    min(y, height - 1)
                )

                x_coordinates.append(x)
                y_coordinates.append(y)


            

            x_min = min(x_coordinates)
            x_max = max(x_coordinates)

            y_min = min(y_coordinates)
            y_max = max(y_coordinates)



            hand_width = x_max - x_min
            hand_height = y_max - y_min


            
            box_size = max(
                hand_width,
                hand_height
            )


            
            box_size += (
                PADDING * 2
            )
            center_x = (
                x_min + x_max
            ) // 2

            center_y = (
                y_min + y_max
            ) // 2


            
            square_x_min = (
                center_x - box_size // 2
            )

            square_y_min = (
                center_y - box_size // 2
            )

            square_x_max = (
                square_x_min + box_size
            )

            square_y_max = (
                square_y_min + box_size
            )
            if square_x_min < 0:

                square_x_max += (
                    -square_x_min
                )

                square_x_min = 0


            if square_y_min < 0:

                square_y_max += (
                    -square_y_min
                )

                square_y_min = 0


            if square_x_max > width:

                shift = (
                    square_x_max - width
                )

                square_x_min -= shift
                square_x_max = width


        
            if square_y_max > height:

                shift = (
                    square_y_max - height
                )

                square_y_min -= shift
                square_y_max = height


            
            square_x_min = max(
                0,
                square_x_min
            )

            square_y_min = max(
                0,
                square_y_min
            )

            square_x_max = min(
                width,
                square_x_max
            )

            square_y_max = min(
                height,
                square_y_max
            )

            hand_crop = image[
                square_y_min:square_y_max,
                square_x_min:square_x_max
            ]


            if hand_crop.size == 0:

                rejected_images += 1
                condition_rejected += 1

                continue



            hand_crop = cv2.resize(
                hand_crop,
                (IMAGE_SIZE, IMAGE_SIZE),
                interpolation=cv2.INTER_AREA
            )



            output_path = os.path.join(
                output_folder,
                filename
            )


            success = cv2.imwrite(
                output_path,
                hand_crop
            )


            if success:

                cleaned_images += 1
                condition_cleaned += 1

            else:

                rejected_images += 1
                condition_rejected += 1


        print(
            f"  {condition}: "
            f"Kept = {condition_cleaned}, "
            f"Rejected = {condition_rejected}"
        )



landmarker.close()



print()
print("============================================")
print("          CLEANING COMPLETE")
print("============================================")

print(
    f"Original images: {total_images}"
)

print(
    f"Cleaned images:  {cleaned_images}"
)

print(
    f"Rejected images: {rejected_images}"
)

print(
    f"Cleaned dataset: {CLEANED_FOLDER}/"
)

print("============================================")