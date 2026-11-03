import os
import cv2
import numpy as np
import tensorflow as tf



MODEL_PATH = "models/asl_clean_model.keras"
CLASSES_PATH = "clean_classes.npy"
DATASET_DIR = "cleaned_dataset"

IMAGE_SIZE = 128
BATCH_SIZE = 32


print("============================================")
print("       TESTING CLEAN ASL MODEL")
print("============================================")



if not os.path.exists(MODEL_PATH):
    print("\nERROR: Model not found:")
    print(MODEL_PATH)
    exit()

if not os.path.exists(CLASSES_PATH):
    print("\nERROR: Class file not found:")
    print(CLASSES_PATH)
    exit()

if not os.path.exists(DATASET_DIR):
    print("\nERROR: Cleaned dataset not found:")
    print(DATASET_DIR)
    exit()


print("\nLoading model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded!")

classes = np.load(
    CLASSES_PATH,
    allow_pickle=True
)

classes = [str(x) for x in classes]

print("Number of classes:", len(classes))

print("\nFinding images...")

image_paths = []
true_labels = []


for class_name in classes:

    class_folder = os.path.join(
        DATASET_DIR,
        class_name
    )

    if not os.path.exists(class_folder):
        continue


    for condition in ["normal", "dark"]:

        condition_folder = os.path.join(
            class_folder,
            condition
        )

        if not os.path.exists(condition_folder):
            continue


        for filename in os.listdir(
            condition_folder
        ):

            if filename.lower().endswith(
                (".jpg", ".jpeg", ".png")
            ):

                image_path = os.path.join(
                    condition_folder,
                    filename
                )

                image_paths.append(
                    image_path
                )

                true_labels.append(
                    class_name
                )


print(
    "Images found:",
    len(image_paths)
)

class_to_index = {
    class_name: index
    for index, class_name
    in enumerate(classes)
}


true_indices = np.array([
    class_to_index[label]
    for label in true_labels
])

print("\nTesting images...")

all_predictions = []

total_images = len(image_paths)


for start in range(
    0,
    total_images,
    BATCH_SIZE
):

    end = min(
        start + BATCH_SIZE,
        total_images
    )


    batch_paths = image_paths[
        start:end
    ]


    batch_images = []



    for image_path in batch_paths:

        image = cv2.imread(
            image_path
        )


        if image is None:

            image = np.zeros(
                (
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                    3
                ),
                dtype=np.uint8
            )

        else:

            image = cv2.resize(
                image,
                (
                    IMAGE_SIZE,
                    IMAGE_SIZE
                )
            )



        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )


        

        image = (
            image.astype("float32")
            / 255.0
        )


        batch_images.append(
            image
        )



    batch_images = np.array(
        batch_images,
        dtype="float32"
    )



    predictions = model.predict(
        batch_images,
        verbose=0
    )


    predicted_indices = np.argmax(
        predictions,
        axis=1
    )


    all_predictions.extend(
        predicted_indices
    )



    if (
        end % 320 == 0
        or end == total_images
    ):

        percentage = (
            end / total_images
        ) * 100

        print(
            f"Tested {end}/{total_images}"
            f" ({percentage:.1f}%)"
        )



predicted_indices = np.array(
    all_predictions
)



correct = np.sum(
    predicted_indices
    == true_indices
)

wrong = total_images - correct

accuracy = (
    correct / total_images
) * 100


print("\n============================================")
print("              RESULTS")
print("============================================")

print(
    f"\nTotal images tested : {total_images}"
)

print(
    f"Correct predictions : {correct}"
)

print(
    f"Wrong predictions   : {wrong}"
)

print(
    f"Accuracy            : {accuracy:.2f}%"
)


print("\n============================================")
print("        CLASS-BY-CLASS ACCURACY")
print("============================================")

print(
    f"{'Class':<8}"
    f"{'Total':<10}"
    f"{'Correct':<10}"
    f"{'Accuracy':<10}"
)

print("-" * 38)


for index, class_name in enumerate(classes):

    mask = (
        true_indices == index
    )

    class_total = np.sum(mask)

    class_correct = np.sum(
        predicted_indices[mask]
        == index
    )


    if class_total > 0:

        class_accuracy = (
            class_correct
            / class_total
        ) * 100

    else:

        class_accuracy = 0


    print(
        f"{class_name:<8}"
        f"{class_total:<10}"
        f"{class_correct:<10}"
        f"{class_accuracy:.2f}%"
    )

print("\n============================================")
print("          TESTING FINISHED")
print("============================================")

