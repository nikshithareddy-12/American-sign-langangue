import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

X_test = np.load("clean_X_test.npy")
y_test = np.load("clean_y_test.npy")
classes = np.load("clean_classes.npy", allow_pickle=True)

model = tf.keras.models.load_model("models/asl_clean_model.keras")

print("\n============================================")
print("           CHECKING MODEL")
print("============================================")


predictions = model.predict(X_test, verbose=1)
y_pred = np.argmax(predictions, axis=1)

accuracy = np.mean(y_pred == y_test)

print(f"\nOverall Test Accuracy: {accuracy * 100:.2f}%")


print("\n============================================")
print("          CLASSIFICATION REPORT")
print("============================================")

report = classification_report(
    y_test,
    y_pred,
    target_names=classes,
    zero_division=0
)

print(report)

cm = confusion_matrix(y_test, y_pred)

print("\n============================================")
print("       MOST COMMON CONFUSIONS")
print("============================================")

confusions = []

for actual in range(len(classes)):
    for predicted in range(len(classes)):
        if actual != predicted and cm[actual][predicted] > 0:
            confusions.append(
                (
                    cm[actual][predicted],
                    classes[actual],
                    classes[predicted]
                )
            )


confusions.sort(reverse=True)

for count, actual, predicted in confusions[:20]:
    print(f"{actual} -> {predicted}: {count} mistakes")

print("\n============================================")
print("              CHECK COMPLETE")
print("============================================")