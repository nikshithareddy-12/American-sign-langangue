import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


DATASET_FOLDER = "dataset"
IMAGE_SIZE = 128




DATASET_FOLDER = "dataset"
IMAGE_SIZE = 128


images = []
labels = []

print("\n========================================")
print("PREPARING DATASET")
print("========================================")




for sign in os.listdir(DATASET_FOLDER):

    sign_folder = os.path.join(DATASET_FOLDER, sign)

    if not os.path.isdir(sign_folder):
        continue

    print(f"Reading: {sign}")


    for condition in ["normal", "dark"]:

        condition_folder = os.path.join(
            sign_folder,
            condition
        )

        if not os.path.exists(condition_folder):
            continue

        for filename in os.listdir(condition_folder):

            if not filename.lower().endswith(".jpg"):
                continue

            image_path = os.path.join(
                condition_folder,
                filename
            )

            image = cv2.imread(image_path)

            if image is None:
                print(f"Could not read: {image_path}")
                continue

            
            image = cv2.resize(
                image,
                (IMAGE_SIZE, IMAGE_SIZE)
            )

            
            image = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )

            images.append(image)
            labels.append(sign)
X = np.array(images)
y = np.array(labels)
print("\nImages loaded:", len(X))
print("Image shape:", X.shape)
print("Labels:", len(y))
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
print("\nClasses:")
print(encoder.classes_)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0
np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)
np.save("classes.npy", encoder.classes_)


print("\n========================================")
print("DATASET PREPARATION COMPLETE!")
print("========================================")

print("Training images:", len(X_train))
print("Testing images:", len(X_test))
print("Number of classes:", len(encoder.classes_))

print("\nSaved files:")
print("X_train.npy")
print("X_test.npy")
print("y_train.npy")
print("y_test.npy")
print("classes.npy")

print("========================================")