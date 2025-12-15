"""Train a 3D DenseNet to predict age from preprocessed T1 NIfTI volumes.

Usage example:
  python IXI_T1_DenseNet.py --csv ixi_full.csv --img_col T1 --age_col Age --n_samples 200 --batch_size 4 --epochs 20 --model_out T1_DenseNet_AgePredictor.h5

Notes:
- Expects input NIfTI volumes of shape 240x240x155 (channel-last). If a
  loaded volume has different shape it will be center-cropped or zero-padded
  to the target shape.
- Uses a tf.data pipeline with a nibabel loader wrapped via tf.numpy_function.
"""

import argparse
import logging
from pathlib import Path
import os
import random
import numpy as np
import pandas as pd
import nibabel as nib
import tensorflow as tf
from sklearn.model_selection import train_test_split


TARGET_SHAPE = (240, 240, 155)


def _parse_args():
    p = argparse.ArgumentParser(description='Train 3D DenseNet on IXI T1 images to predict age')
    p.add_argument('--csv', default='ixi_combined.csv', help='CSV with image paths and Age')
    p.add_argument('--img_col', default='T1', help='column name with T1 image paths')
    p.add_argument('--age_col', default='Age', help='column name with Age')
    p.add_argument('--n_samples', type=int, default=None, help='number of samples to use (random sample); default uses all')
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--val_split', type=float, default=0.2)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--model_out', default='T1_DenseNet_AgePredictor.h5')
    p.add_argument('--verbose', action='store_true')
    return p.parse_args()


def center_crop_or_pad(img: np.ndarray, target: tuple) -> np.ndarray:
    """Center-crop or pad a 3D array to target shape.

    img: (H, W, D)
    target: (H_t, W_t, D_t)
    """
    out = np.zeros((*target,), dtype=img.dtype)
    in_shape = img.shape
    # compute start indices for cropping or padding
    starts = [max((in_shape[i] - target[i]) // 2, 0) for i in range(3)]
    copies = [min(in_shape[i], target[i]) for i in range(3)]

    # compute destination start in out
    dest_starts = [max((target[i] - in_shape[i]) // 2, 0) for i in range(3)]

    src_slices = tuple(slice(starts[i], starts[i] + copies[i]) for i in range(3))
    dst_slices = tuple(slice(dest_starts[i], dest_starts[i] + copies[i]) for i in range(3))
    out[dst_slices] = img[src_slices]
    return out


def load_nifti_numpy(path: str) -> np.ndarray:
    img = nib.load(path).get_fdata().astype(np.float32)
    # if 4D take first volume
    if img.ndim == 4:
        img = img[..., 0]
    # normalize intensities
    mn = np.min(img)
    mx = np.max(img)
    if mx - mn > 0:
        img = (img - mn) / (mx - mn)
    else:
        img = img * 0.0

    img = center_crop_or_pad(img, TARGET_SHAPE)
    # add channel dim
    img = img[..., np.newaxis]
    return img.astype(np.float32)


def tf_loader(path, label):
    def _load(path_str):
        try:
            arr = load_nifti_numpy(path_str.decode())
        except Exception as e:
            print(f"Error loading {path_str}: {e}")
            # return zeros to keep shapes consistent; training will ignore if samples filtered earlier
            arr = np.zeros((*TARGET_SHAPE, 1), dtype=np.float32)
        return arr

    img = tf.numpy_function(_load, [path], tf.float32)
    img.set_shape((*TARGET_SHAPE, 1))
    label = tf.cast(label, tf.float32)
    return img, label


def build_densenet(input_shape=TARGET_SHAPE + (1,), growth_rate=32, block_layers=(6, 12, 24, 16), compression=0.5):
    """Build a 3D DenseNet-121-like model.

    block_layers: tuple specifying number of bottleneck layers per dense block
    growth_rate: number of filters to add per dense layer
    compression: reduction factor at transition layers
    """
    inputs = tf.keras.Input(shape=input_shape)

    # Initial convolution and pooling
    x = tf.keras.layers.Conv3D(64, kernel_size=7, strides=2, padding='same', use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling3D(pool_size=3, strides=2, padding='same')(x)

    def dense_layer(x, growth_rate):
        # Bottleneck layer
        bn1 = tf.keras.layers.BatchNormalization()(x)
        relu1 = tf.keras.layers.Activation('relu')(bn1)
        inter_filters = 4 * growth_rate
        conv1 = tf.keras.layers.Conv3D(inter_filters, kernel_size=1, strides=1, padding='same', use_bias=False)(relu1)

        bn2 = tf.keras.layers.BatchNormalization()(conv1)
        relu2 = tf.keras.layers.Activation('relu')(bn2)
        conv2 = tf.keras.layers.Conv3D(growth_rate, kernel_size=3, strides=1, padding='same', use_bias=False)(relu2)

        x_out = tf.keras.layers.Concatenate()([x, conv2])
        return x_out

    def dense_block(x, num_layers, growth_rate):
        for _ in range(num_layers):
            x = dense_layer(x, growth_rate)
        return x

    def transition_layer(x, compression):
        bn = tf.keras.layers.BatchNormalization()(x)
        relu = tf.keras.layers.Activation('relu')(bn)
        out_filters = int(tf.keras.backend.int_shape(x)[-1] * compression)
        conv = tf.keras.layers.Conv3D(out_filters, kernel_size=1, strides=1, padding='same', use_bias=False)(relu)
        x_out = tf.keras.layers.AveragePooling3D(pool_size=2, strides=2, padding='same')(conv)
        return x_out

    # Build dense blocks with transitions
    num_filters = 64
    x = x
    for i, layers_in_block in enumerate(block_layers):
        x = dense_block(x, layers_in_block, growth_rate)
        num_filters += layers_in_block * growth_rate
        # add transition after all but the last block
        if i != len(block_layers) - 1:
            x = transition_layer(x, compression)
            num_filters = int(num_filters * compression)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.GlobalAveragePooling3D()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(1, activation='linear')(x)

    model = tf.keras.Model(inputs, outputs, name='DenseNet3D-121')
    return model


def main():
    args = _parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
    else:
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    df = pd.read_csv(args.csv) if args.csv.endswith('.csv') else pd.read_excel(args.csv)
    if args.img_col not in df.columns or args.age_col not in df.columns:
        raise ValueError(f"CSV must contain columns: {args.img_col} and {args.age_col}")

    # Filter out missing entries and non-existing files
    df = df.dropna(subset=[args.img_col, args.age_col])
    df[args.img_col] = df[args.img_col].astype(str)
    df['exists'] = df[args.img_col].apply(lambda p: Path(p).exists())
    df = df[df['exists']]
    df = df.reset_index(drop=True)

    if df.empty:
        raise RuntimeError('No valid samples found after filtering missing files/ages')

    # sample limit
    if args.n_samples and args.n_samples < len(df):
        df = df.sample(n=args.n_samples, random_state=args.seed).reset_index(drop=True)

    paths = df[args.img_col].tolist()
    ages = df[args.age_col].astype(float).tolist()

    X_train_p, X_val_p, y_train, y_val = train_test_split(paths, ages, test_size=args.val_split, random_state=args.seed)

    train_ds = tf.data.Dataset.from_tensor_slices((X_train_p, y_train))
    val_ds = tf.data.Dataset.from_tensor_slices((X_val_p, y_val))

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.shuffle(buffer_size=1000, seed=args.seed)
    train_ds = train_ds.map(lambda p, l: tf_loader(p, l), num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.batch(args.batch_size).prefetch(AUTOTUNE)

    val_ds = val_ds.map(lambda p, l: tf_loader(p, l), num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.batch(args.batch_size).prefetch(AUTOTUNE)

    model = build_densenet()
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='mean_squared_error', metrics=[tf.keras.metrics.MeanAbsoluteError()])
    model.summary()

    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs)

    model.save(args.model_out)
    logging.info('Saved model to %s', args.model_out)


if __name__ == '__main__':
    main()
