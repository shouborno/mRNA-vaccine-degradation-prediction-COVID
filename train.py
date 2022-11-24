import os
import subprocess
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import (
    LSTM,
    BatchNormalization,
    Dense,
    LeakyReLU,
    SpatialDropout1D,
    TimeDistributed,
)
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from keras_layer_normalization import LayerNormalization
from loguru import logger
from sklearn.utils import shuffle
from tensorflow.keras import backend as K
from tqdm import tqdm

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
dataset_directory = os.path.join(__location__, "data")


def download_dataset():
    dataset_zip = "stanford-covid-vaccine.zip"
    if not os.path.isfile(dataset_zip):
        download_command = "kaggle competitions download -c stanford-covid-vaccine"
        try:
            subprocess.run(download_command.split(), check=True)
        except subprocess.CalledProcessError:
            download_failed_message = (
                "Failed to download the dataset. "
                "Follow https://github.com/Kaggle/kaggle-api#api-credentials"
            )
            logger.error(download_failed_message)
            raise FileNotFoundError(download_failed_message)

    train_file = os.path.join(dataset_directory, "train.json")

    if not os.path.isfile(train_file):
        os.makedirs(dataset_directory, exist_ok=True)
        with zipfile.ZipFile(dataset_zip, "r") as zip_ref:
            zip_ref.extractall(dataset_directory)

    train = pd.read_json(train_file, lines=True)
    logger.info(f"Data shape: {train.shape}")
    logger.info(f"Data example:\n{train.head()}")

    return train


def prepare_dataset():
    train = download_dataset()

    # Byte Pairing Probability matrix
    bpps_train = []
    for id in tqdm(train["id"]):
        bpp = np.load(os.path.join(dataset_directory, f"bpps/{id}.npy"))
        bpps_train.append(bpp)
    bpps_train = np.array(bpps_train)

    # Nitrogenous base sequence
    base_dictionary = {"A": 0, "C": 1, "G": 2, "U": 3}
    base_train = []

    for sample in tqdm(train["sequence"]):
        sequence = []
        for base in sample:
            sequence.append(base_dictionary[base])
        base_train.append(to_categorical(sequence, num_classes=len(base_dictionary)))

    base_train = np.array(base_train)

    # Predicted loop type (bpRNA)
    loop_type_dictionary = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}

    loop_train = []

    for sample in tqdm(train["predicted_loop_type"]):
        sample_loop_type = []
        for loop_type in sample:
            sample_loop_type.append(loop_type_dictionary[loop_type])
        loop_train.append(to_categorical(sample_loop_type, num_classes=len(loop_type_dictionary)))

    loop_train = np.array(loop_train)

    # Structure (pairs represented by brackets, non-pairs represented by dots)
    structure_dictionary = {".": 0, "(": 1, ")": 2}
    structure_train = []

    for sample in tqdm(train["structure"]):
        a = []
        for structure in sample:
            a.append(structure_dictionary[structure])
        structure_train.append(to_categorical(a, num_classes=len(structure_dictionary)))

    structure_train = np.array(structure_train)

    # Limiting to 91 bases due to Stanford's measurement limitations
    train_x = np.concatenate(
        (
            base_train[:, :91, :],
            loop_train[:, :91, :],
            structure_train[:, :91, :],
            bpps_train[:, :91, :91],
        ),
        axis=2,
    )
    logger.debug(f"Training data shape: {train_x.shape}")

    target_sample = pd.read_csv(os.path.join(dataset_directory, "sample_submission.csv"))
    targets = train[target_sample.columns[1:]].values
    train_y = []
    for y in tqdm(targets):
        train_y.append(np.array(list(y)).T)
    train_y = np.array(train_y)

    """
    Padding train_y with 0s to match the (sample, 91, features) shape,
    since only 68 bases are labeled for the training set.
    """
    logger.debug(f"Training target shape before reshaping: {train_y.shape}")
    padded_train_y = []
    for y in train_y:
        padded_train_y.append(
            np.vstack((y, np.zeros((train_x.shape[1] - train_y.shape[1], train_y.shape[2]))))
        )
    train_y = np.array(padded_train_y)
    logger.debug(f"Training target shape after reshaping: {train_y.shape}")

    logger.info(f"Target values range from {np.min(train_y)} to {np.max(train_y)}")

    return train_x, train_y


def rmse(y_actual, y_pred):
    mse = tf.keras.losses.mean_squared_error(y_actual, y_pred)
    return K.sqrt(mse)


def mcrmse_loss(y_actual, y_pred, num_scored=5):
    y_actual = y_actual[:, :68, :]
    y_pred = y_pred[:, :68, :]
    score = 0
    for i in range(num_scored):
        score += rmse(y_actual[:, :, i], y_pred[:, :, i]) / num_scored
    return score


def baseline_model(x, y):
    model = Sequential()
    model.add(
        LSTM(
            5,
            activation="linear",
            input_shape=(x.shape[1:]),
            return_sequences=True,
            recurrent_dropout=0.5,
        )
    )

    model.add(LSTM(50, activation="tanh", return_sequences=True))
    model.add(LSTM(250, activation="tanh", return_sequences=True, recurrent_dropout=0.5))
    model.add(LayerNormalization())
    model.add(SpatialDropout1D(0.5))
    model.add(LSTM(50, activation="tanh", return_sequences=True, recurrent_dropout=0.5))
    model.add(LSTM(5, activation="tanh", return_sequences=True, recurrent_dropout=0.5))
    model.add(BatchNormalization())
    model.add(TimeDistributed(Dense(y.shape[2], activation=LeakyReLU())))

    model.compile(optimizer="rmsprop", loss=mcrmse_loss)
    return model


def train():
    train_x, train_y = prepare_dataset()
    x, y = shuffle(train_x.astype("float32"), train_y.astype("float32"))
    model = baseline_model(x, y)
    model.summary()

    mcp_save = ModelCheckpoint(
        "best.hdf5", save_best_only=True, monitor="val_loss", mode="min", verbose=1
    )

    tensorboard = TensorBoard(log_dir="./", histogram_freq=1, write_images=True)
    mcp_save = ModelCheckpoint(
        "best.hdf5", save_best_only=True, monitor="val_loss", mode="min", verbose=1
    )
    history = model.fit(
        x,
        y,
        epochs=2,
        validation_split=0.33,
        batch_size=128,
        verbose=1,
        shuffle=True,
        callbacks=[mcp_save, tensorboard],
    )

    # summarize history for loss
    plt.figure(figsize=(16, 12))
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "val"], loc="upper left")
    plt.show()


if __name__ == "__main__":
    train()
