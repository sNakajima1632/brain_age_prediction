"""
IXI_T1densenet.py
Train a DenseNet-like 3D CNN to predict subject age from preprocessed T1-weighted MRI volumes.

Folder structure:
age prediction/
|-- data.xlsx
|-- IXI_T1densenet.py
|-- IXI_preprocessed/
|   |-- <subject_id>/
|       |-- <subject_id>_T1.nii.gz
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
import tensorflow as tf
from tensorflow import keras
from keras import layers, models, optimizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ============ Configuration ============
DATA_XLSX = "IXI_updated.xls"
DATA_DIR = "IXIprep"
IMG_TYPE = "_SRI_t1.nii.gz"     # Only T1-weighted
IMG_DIMS = (128, 128, 128)  # Resize to standard dimensions
BATCH_SIZE = 4
EPOCHS = 20
VAL_SPLIT = 0.2
SEED = 42

# ============ Utility Functions ============
def load_nifti_image(filepath):
    """Load and normalize a NIfTI MRI volume."""
    img = nib.load(filepath)
    data = img.get_fdata().astype(np.float32)
    # Normalize intensities between 0–1
    data = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-5)
    # Resize to IMG_DIMS
    data = tf.image.resize(tf.convert_to_tensor(data[..., np.newaxis]), IMG_DIMS[:2]).numpy()
    # Pad/crop along depth if needed
    if data.shape[2] > IMG_DIMS[2]:
        data = data[:, :, :IMG_DIMS[2], :]
    elif data.shape[2] < IMG_DIMS[2]:
        pad = IMG_DIMS[2] - data.shape[2]
        data = np.pad(data, ((0, 0), (0, 0), (0, pad), (0, 0)), mode='constant')
    return data

def create_dataset(xlsx_path, data_dir, img_type=IMG_TYPE):
    """Load MRI data and ages from spreadsheet."""
    df = pd.read_excel(xlsx_path)
    X, y = [], []

    for _, row in df.iterrows():
        subj_id, age = str(row['subject_id']), float(row['age'])
        nifti_path = os.path.join(data_dir, subj_id, f"{subj_id}{img_type}")
        if os.path.exists(nifti_path):
            img = load_nifti_image(nifti_path)
            X.append(img)
            y.append(age)
        else:
            print(f"Warning: Missing {nifti_path}")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    print(f"Loaded {len(X)} images.")
    return X, y

# ============ Build DenseNet Model ============
def dense_block(x, num_convs, growth_rate):
    for _ in range(num_convs):
        bn = layers.BatchNormalization()(x)
        relu = layers.Activation('relu')(bn)
        conv = layers.Conv3D(growth_rate, kernel_size=3, padding='same')(relu)
        x = layers.Concatenate()([x, conv])
    return x

def transition_layer(x, reduction):
    bn = layers.BatchNormalization()(x)
    conv = layers.Conv3D(int(tf.keras.backend.int_shape(x)[-1] * reduction), 1)(bn)
    avg = layers.AveragePooling3D(2, strides=2)(conv)
    return avg

def build_densenet(input_shape=IMG_DIMS + (1,), growth_rate=16, num_blocks=3, num_layers_per_block=3):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv3D(32, 3, padding="same", activation="relu")(inputs)
    x = layers.MaxPooling3D(2)(x)

    for _ in range(num_blocks):
        x = dense_block(x, num_layers_per_block, growth_rate)
        x = transition_layer(x, 0.5)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="linear")(x)

    model = keras.Model(inputs, outputs, name="DenseNet3D_AgePrediction")
    return model

# ============ Main ============
if __name__ == "__main__":
    # Load dataset
    X, y = create_dataset(DATA_XLSX, DATA_DIR)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=VAL_SPLIT, random_state=SEED)

    # Build model
    model = build_densenet()
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss="mean_squared_error",
        metrics=["mae"]
    )

    model.summary()

    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )

    # Save model
    model.save("T1_DenseNet_AgePredictor.h5")
    print("Model saved as T1_DenseNet_AgePredictor.h5")
