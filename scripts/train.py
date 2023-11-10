import os
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from seaborn import heatmap
from tensorflow import keras
import segmentation_models as sm
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix


from argparse import ArgumentParser
from datagenerator import MyGenerator

AVAILABLE_MODELS = [
    "vgg16",
    "vgg19",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "seresnet18",
    "efficientnetb0",
    "efficientnetb1",
    "efficientnetb2",
    "efficientnetb6",
]

AVAILABLE_ARCHITECTURES = ["unet", "linknet"]

# Init Argument Parser
parser = ArgumentParser()
parser.add_argument(
    "--x-data", "-xd", type=str, required=True, help="x_data folder path"
)
parser.add_argument(
    "--y-data", "-yd", type=str, required=True, help="y_data folder path"
)
parser.add_argument(
    "--x-data-t",
    "-xdt",
    type=str,
    required=True,
    help="x_data folder path for test/validation",
)
parser.add_argument(
    "--y-data-t",
    "-ydt",
    type=str,
    required=True,
    help="y_data folder path for test/validation",
)
parser.add_argument(
    "--backbone",
    type=str,
    required=True,
    help="Backbone for model",
    choices=AVAILABLE_MODELS,
)
parser.add_argument("--output", "-o", type=str, help="Output path", default="OUTPUT")
parser.add_argument("--epochs", "-e", type=int, help="Number of epochs", default=25)
parser.add_argument(
    "--architecture",
    "-a",
    type=str,
    help="Architecture to train. Default U-Net",
    choices=AVAILABLE_ARCHITECTURES,
    default=AVAILABLE_ARCHITECTURES[0],
)
args = parser.parse_args()

BACKBONE = args.backbone
ARCHITECTURE = args.architecture
X_DATA = args.x_data
Y_DATA = args.y_data
if not (Path(X_DATA).exists() and Path(Y_DATA).exists()):
    raise FileNotFoundError("No x_data and y_data folder found")

X_DATA_T = args.x_data_t
Y_DATA_T = args.y_data_t
if not (Path(X_DATA_T).exists() and Path(Y_DATA_T).exists()):
    raise FileNotFoundError("No x_data and y_data folder for test/validation found")

OUTPUT = Path(args.output)
if not OUTPUT.exists():
    os.mkdir(str(OUTPUT))

EPOCHS = args.epochs
if EPOCHS <= 0:
    raise ValueError("Epochs must be positive integer")

# Init generators
train_generator = MyGenerator(X_DATA, Y_DATA)
test_generator = MyGenerator(X_DATA_T, Y_DATA_T, split=0.8)
val_generator = MyGenerator(X_DATA_T, Y_DATA_T, split=-0.2)


# Create model
if ARCHITECTURE == "unet":
    model = sm.Unet(
        BACKBONE,
        input_shape=(None, None, 2),
        encoder_weights=None,
        classes=5,
        activation="softmax",
    )
elif ARCHITECTURE == "linknet":
    model = sm.Linknet(
        BACKBONE,
        input_shape=(None, None, 2),
        encoder_weights=None,
        classes=5,
        activation="softmax",
    )
model.compile("adam", "categorical_crossentropy", ["accuracy"])

# Create callbacks
callbacks = [
    keras.callbacks.ModelCheckpoint(
        str(
            OUTPUT
            / (
                "{epoch:02d}-{val_loss:.4f}val_loss-{val_accuracy:.4f}%-"
                + ARCHITECTURE
                + "-"
                + BACKBONE
                + ".h5"
            )
        ),
        monitor="val_loss",
        save_best_only=True,
    ),
    keras.callbacks.EarlyStopping(patience=10, monitor="val_loss"),
    keras.callbacks.ReduceLROnPlateau(patience=3, monitor="val_loss"),
]

# Train model
history = model.fit(
    train_generator, epochs=EPOCHS, callbacks=callbacks, validation_data=test_generator
)

with open(str(OUTPUT / (ARCHITECTURE + "-" + BACKBONE + ".csv")), "w") as f:
    pd.DataFrame(history.history).to_csv(f)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], "g")
plt.plot(history.history["val_accuracy"], "r")
plt.legend(["Train accuracy", "Validation accuracy"])
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title(BACKBONE + " accuracy")

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], "g")
plt.plot(history.history["val_loss"], "r")
plt.legend(["Train loss", "Validation loss"])
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title(BACKBONE + " loss")
plt.savefig(str(OUTPUT / (ARCHITECTURE + "-" + BACKBONE + ".png")))


# Calc F1 score
y = []
for el in val_generator:
    y.append(el[1])
y = np.array(y)
y = y.argmax(axis=-1)
y.shape = (-1, 1)

pred = model.predict(val_generator, batch_size=64).argmax(axis=-1)
pred.shape = (-1, 1)

cr = classification_report(
    y,
    pred,
    target_names=["no data", "deforestation", "forestation", "no change", "water"],
)

cm = confusion_matrix(y, pred)

print(cr)
print()
print(cm)

cm = cm / y.size

cm_copy = cm.copy()
for i in range(5):
    cm_copy[i][i] = 0

plt.subplot(1, 1, 1)
plt.figure(figsize=(10, 10))
hm = heatmap(
    cm_copy,
    xticklabels=["no data", "deforestation", "forestation", "no change", "water"],
    yticklabels=["no data", "deforestation", "forestation", "no change", "water"],
)
hm.set_xlabel("Predicted")
hm.set_ylabel("Actual")
plt.savefig(
    str(OUTPUT / (ARCHITECTURE + "-" + BACKBONE + "-confusion-matrix-no-middle.png"))
)
