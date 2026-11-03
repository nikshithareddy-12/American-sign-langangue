import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

print("============================================")
print("       TRAINING CLEAN ASL MODEL")
print("============================================")


X_train = np.load("clean_X_train.npy")
X_validation = np.load("clean_X_validation.npy")
X_test = np.load("clean_X_test.npy")

y_train = np.load("clean_y_train.npy")
y_validation = np.load("clean_y_validation.npy")
y_test = np.load("clean_y_test.npy")

classes = np.load("clean_classes.npy", allow_pickle=True)

print("\nDataset loaded:")
print("Training:", X_train.shape)
print("Validation:", X_validation.shape)
print("Testing:", X_test.shape)
print("Classes:", len(classes))


data_augmentation = tf.keras.Sequential([
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.10),
    layers.RandomTranslation(0.10, 0.10),
], name="data_augmentation")



model = models.Sequential([

    layers.Input(shape=(128, 128, 3)),

    
    data_augmentation,

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    
    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    
    layers.Conv2D(256, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),


    layers.Dense(256, activation="relu"),


    layers.Dropout(0.5),

    
    layers.Dense(len(classes), activation="softmax")
])



model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nModel created successfully.")

model.summary()



early_stopping = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)


print("\n============================================")
print("              STARTING TRAINING")
print("============================================")

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_validation, y_validation),
    epochs=30,
    batch_size=32,
    callbacks=[early_stopping],
    shuffle=True
)



print("\n============================================")
print("              TEST RESULTS")
print("============================================")

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=1
)

print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")



model.save("models/asl_clean_model.keras")

print("\n============================================")
print("          MODEL SAVED SUCCESSFULLY")
print("============================================")

print("Saved as:")
print("models/asl_clean_model.keras")