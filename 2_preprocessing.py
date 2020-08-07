from utils import DataModule

import numpy as np
import cv2


dm = DataModule("None")


file_name = input("file name?: ")
nploader = np.load(f"ARC/{file_name}.npz")
splited = bool(int(input("Is it splited?\n(0 or 1): ")))
augmented = bool(int(input("Is it augmented?\n(0 or 1): ")))

count = 0

if splited is True:
    dataset_dir_path = "dataset/"

    train_img_path, train_label = nploader["train_img_path"], nploader["train_label"]
    test_img_path, test_label = nploader["test_img_path"], nploader["test_label"]

    train_label = np.array(train_label, dtype=np.float)
    test_label = np.array(test_label, dtype=np.float)

    x_train = list()
    x_test = list()

    for path in train_img_path:
        img_path = dataset_dir_path + "train/" + path
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, dsize=(400, 300))
        x_train.append(img)
        if augmented is True:
            img_2 = cv2.flip(img, flipCode=1)
            x_train.append(img_2)

    for path in test_img_path:
        img_path = dataset_dir_path + "test/" + path
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, dsize=(400, 300))
        x_test.append(img)
        if augmented is True:
            img_2 = cv2.flip(img, flipCode=1)
            x_test.append(img_2)

    print(np.shape(x_train), np.shape(train_label), np.shape(x_test), np.shape(test_label))
    np.savez_compressed(f"APP/{file_name}_(300, 400).npz", x_train=x_train, x_test=x_test, train_label=train_label, test_label=test_label)

elif splited is False:
    dataset_dir_path = "dataset/"

    img_path, label = nploader["img_path"], nploader["label"]

    label = np.array(label, dtype=np.int)

    x_data = list()

    for path in img_path:
        img_path = dataset_dir_path + "all/" + path
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = dm.image_resize(img)      # resize to width: 500; height: 400
        x_data.append(img)
        if augmented is True:
            img_2 = cv2.flip(img, flipCode=1)
            x_data.append(img_2)

    print(np.shape(x_data), np.shape(label))
    np.savez_compressed(f"APP/{file_name}_(400, 500).npz", x_data=x_data, label=label)
