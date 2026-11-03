import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models




X_train = np.load("X_train.npy")
X_test = np.load("X_test.npy")

y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")

classes = np.load("classes.npy", allow_pickle=True)


print("Training images:", len(X_train))
print("Testing images:", len(X_test))
print("Number of classes:", len(classes))




model = models.Sequential([

    layers.Input(shape=(128, 128, 3)),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),

    layers.Dense(128, activation="relu"),

    layers.Dropout(0.5),

    layers.Dense(len(classes), activation="softmax")
])




model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
model.summary()
history = model.fit(
    X_train,
    y_train,
    validation_split=0.1,
    epochs=10,
    batch_size=32,
    shuffle=True
)
test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test
)

print("\n========================================")
print("MODEL TEST RESULTS")
print("========================================")
print("Test Accuracy:", test_accuracy)
print("Test Loss:", test_loss)


os.makedirs("models", exist_ok=True)

model.save("models/asl_model.keras")

print("\nModel saved successfully!")
print("Location: models/asl_model.keras")