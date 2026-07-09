import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam


TRAIN_DIR = "datasets/facial_emotion/archive (14)/train"
TEST_DIR = "datasets/facial_emotion/archive (14)/test"

MODEL_DIR = "facial_emotion/model"
MODEL_PATH = os.path.join(MODEL_DIR, "face_emotion_model.h5")

IMG_SIZE = 48
BATCH_SIZE = 64
EPOCHS = 5


print("Loading FER2013 dataset...")


train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)


train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)


print("\nBuilding CNN model...")


model = Sequential()

model.add(
    Conv2D(
        32,
        (3, 3),
        activation="relu",
        input_shape=(48, 48, 1)
    )
)

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(
    Conv2D(
        64,
        (3, 3),
        activation="relu"
    )
)

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(
    Conv2D(
        128,
        (3, 3),
        activation="relu"
    )
)

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(256, activation="relu"))

model.add(Dropout(0.5))

model.add(
    Dense(
        train_generator.num_classes,
        activation="softmax"
    )
)


model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


print("\nTraining facial emotion model...\n")


history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=EPOCHS
)


os.makedirs(MODEL_DIR, exist_ok=True)

model.save(MODEL_PATH)

print("\nFacial emotion model saved successfully.")

