import os
import pickle
import tarfile as tar
from pathlib import Path
from random import choice
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def extract_tarfile(tarfile_path: str) -> Path:
    tarfile_path = Path(tarfile_path)

    if not tarfile_path.exists():
        raise FileNotFoundError

    with tar.open(tarfile_path, "r:gz") as extracted_file:
        extracted_file_dir = tarfile_path.parent / tarfile_path.stem.split(".tar")[0]
        if extracted_file_dir.is_dir():
            print(f"Path {extracted_file_dir} already exists.")
        else:
            print(f"Creating Path {extracted_file_dir}.")
            Path.mkdir(extracted_file_dir)

        print(
            f"Extracting all the contents of the {tarfile_path} file into {extracted_file_dir}"
        )
        extracted_file.extractall(extracted_file_dir)
    return extracted_file_dir


def get_class_names(root_dir: str | Path) -> Tuple[List[str], Dict[int, str]]:
    root_dir = Path(root_dir)
    class_file_path = sorted(list(root_dir.glob("*/*.meta")))[0]
    with open(class_file_path, "rb") as classes_file:
        classes = pickle.load(classes_file)
        classes = classes["label_names"]
        classes_to_idx = {i: c for i, c in enumerate(classes)}
        return classes, classes_to_idx


def get_classes(root_dir: str) -> Tuple[List[str], Dict[str, int]]:
    class_names = sorted(
        [entry.name for entry in list(os.scandir(root_dir)) if entry.is_dir()]
    )
    if not class_names:  # checks if class_names list is empty
        raise FileNotFoundError(
            f"Couldn't find any classes in {root_dir}. Check File Structure."
        )
    class_to_idx = {s: i for i, s in enumerate(class_names)}
    return class_names, class_to_idx


def deserialize_data(root_dir):
    with open(root_dir, "rb") as train_batch:
        train_dict = pickle.load(train_batch, encoding="latin1")
        del train_dict["batch_label"]
    return train_dict


def preprocess_image_data(image_array: List) -> np.ndarray:
    return np.array(image_array).reshape(3, 32, 32).transpose(1, 2, 0)


def save_images(directory: Path, batch_paths: List[Path], index_to_classes: Dict[int, str]) -> None:
    for batch_path in batch_paths:
        train_batch_dict = deserialize_data(batch_path)
        images = train_batch_dict["data"]
        labels = train_batch_dict["labels"]
        filenames = train_batch_dict["filenames"]
        for index in range(0, len(images)):
            class_name = index_to_classes.get(labels[index])
            image_store_path = directory / class_name / filenames[index]
            if not image_store_path.parent.is_dir():
                Path.mkdir(image_store_path.parent)
            image_array = preprocess_image_data(images[index])
            Image.fromarray(image_array).save(image_store_path)
            # plt.imsave(
            #     image_store_path,
            #     arr=image_array,
            # )


def read_and_save_images(root_dir: str) -> None:
    root_dir = Path(root_dir)
    train_dir = root_dir / "train"
    test_dir = root_dir / "test"

    _, index_to_classes = get_class_names(root_dir)
    print(index_to_classes)

    if not train_dir.is_dir():
        print(f"Creating 'train' directory at the path {train_dir}")
        Path.mkdir(train_dir)
    else:
        print(f"'train' directory already present at path {train_dir}")

    if not test_dir.is_dir():
        print(f"Creating 'test' directory at the path {test_dir}")
        Path.mkdir(test_dir)
    else:
        print(f"'test' directory already present at path {train_dir}")

    training_batch_paths = list(root_dir.glob("*/data*"))
    save_images(directory=train_dir, batch_paths=training_batch_paths, index_to_classes=index_to_classes)

    test_batch_paths = list(root_dir.glob("*/test*"))
    save_images(directory=test_dir, batch_paths=test_batch_paths, index_to_classes=index_to_classes)


def visualise_images(directory: str, nrows: int = 4, ncols: int = 4) -> None:
    directory = Path(directory)
    figure = plt.figure(figsize=(8, 8))
    image_path_list = list(directory.glob("*/*/*.png"))
    for subplot_index in range(1, nrows * ncols + 1):
        image_path = choice(image_path_list)
        image_class = image_path.parent.stem
        image = Image.open(image_path)
        figure.add_subplot(nrows, ncols, subplot_index)
        plt.title(image_class)
        plt.imshow(image)
        plt.tight_layout()
        plt.axis(False)


def plot_loss_curves(results):
    loss = results["train_loss"]
    test_loss = results["test_loss"]

    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]

    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()
