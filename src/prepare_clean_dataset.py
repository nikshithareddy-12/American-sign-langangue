import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder

DATASET_DIR = "cleaned_dataset"
IMAGE_SIZE = 128

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

print("============================================")
print("   PREPARING CLEANED ASL DATASET")
print("============================================")

# Check that ratios are correct
if TRAIN_RATIO + VALIDATION_RATIO + TEST_RATIO != 1.0:
    raise ValueError("Train/Validation/Test ratios must add up to 1.0")


classes = sorted([
    folder for folder in os.listdir(DATASET_DIR)
    if os.path.isdir(os.path.join(DATASET_DIR, folder))
])

print(f"\nClasses found: {len(classes)}")
print(classes)



train_images = []
train_labels = []

validation_images = []
validation_labels = []

test_images = []
test_labels = []


for class_name in classes:

    print(f"\nProcessing: {class_name}")

    class_path = os.path.join(DATASET_DIR, class_name)

    all_images = []


    for condition in ["normal", "dark"]:

        condition_path = os.path.join(class_path, condition)

        if not os.path.exists(condition_path):
            continue

        for filename in os.listdir(condition_path):

            if filename.lower().endswith(
                (".jpg", ".jpeg", ".png")
            ):
                image_path = os.path.join(
                    condition_path,
                    filename
                )

                all_images.append(image_path)

    rng = np.random.default_rng(42)

    rng.shuffle(all_images)

    total = len(all_images)

    

    train_end = int(total * TRAIN_RATIO)

    validation_end = train_end + int(
        total * VALIDATION_RATIO
    )

    train_paths = all_images[:train_end]

    validation_paths = all_images[
        train_end:validation_end
    ]

    test_paths = all_images[
        validation_end:
    ]

    print(
        f"  Total: {total} | "
        f"Train: {len(train_paths)} | "
        f"Validation: {len(validation_paths)} | "
        f"Test: {len(test_paths)}"
    )


    def load_images(image_paths, image_list, label_list):

        for image_path in image_paths:

            image = cv2.imread(image_path)

            if image is None:
                print(
                    f"  WARNING: Could not read {image_path}"
                )
                continue

            image = cv2.resize(
                image,
                (IMAGE_SIZE, IMAGE_SIZE)
            )

            
            image = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )

            
            image = image.astype(
                np.float32
            ) / 255.0

            image_list.append(image)
            label_list.append(class_name)

    

    load_images(
        train_paths,
        train_images,
        train_labels
    )

    load_images(
        validation_paths,
        validation_images,
        validation_labels
    )

    load_images(
        test_paths,
        test_images,
        test_labels
    )



X_train = np.array(train_images, dtype=np.float32)
X_validation = np.array(
    validation_images,
    dtype=np.float32
)
X_test = np.array(test_images, dtype=np.float32)


label_encoder = LabelEncoder()

all_labels = classes

label_encoder.fit(all_labels)

y_train = label_encoder.transform(train_labels)
y_validation = label_encoder.transform(
    validation_labels
)
y_test = label_encoder.transform(test_labels)


np.save("clean_X_train.npy", X_train)
np.save(
    "clean_X_validation.npy",
    X_validation
)
np.save("clean_X_test.npy", X_test)

np.save("clean_y_train.npy", y_train)
np.save(
    "clean_y_validation.npy",
    y_validation
)
np.save("clean_y_test.npy", y_test)

np.save(
    "clean_classes.npy",
    label_encoder.classes_
)


print("\n============================================")
print("       DATA PREPARATION COMPLETE")
print("============================================")

print(f"Training images:    {len(X_train)}")
print(f"Validation images:  {len(X_validation)}")
print(f"Testing images:     {len(X_test)}")

print(
    f"Total images:       "
    f"{len(X_train) + len(X_validation) + len(X_test)}"
)

print(
    f"Number of classes:  "
    f"{len(classes)}"
)

print("\nFiles created:")

print("  clean_X_train.npy")
print("  clean_X_validation.npy")
print("  clean_X_test.npy")
print("  clean_y_train.npy")
print("  clean_y_validation.npy")
print("  clean_y_test.npy")
print("  clean_classes.npy")

print("============================================")