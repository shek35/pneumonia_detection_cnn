import os
import cv2
import numpy as np

def loading_preprocessing(path):
    img_list = []
    for img_path in os.listdir(path):
        img = cv2.imread(path+'/'+img_path)
        if img_path.startswith('.'):  # Skip hidden files like .DS_Store
            continue
        if img is None:  # Skip if image is not loaded
            print(f"Image {img_path} could not be loaded.")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128,128))
        img = img / 128.0
        img_list.append(img)
    return img_list


def load_data():
    train_class1 = loading_preprocessing('data/chest_xray/train/NORMAL')
    train_class2 = loading_preprocessing('data/chest_xray/train/PNEUMONIA')
    test_class1 = loading_preprocessing('data/chest_xray/test/NORMAL')
    test_class2 = loading_preprocessing('data/chest_xray/test/PNEUMONIA')

    train_data = train_class1 + train_class2
    train_labels = [0] * len(train_class1) + [1] * len(train_class2)
    test_data = test_class1 + test_class2
    test_labels = [0] * len(test_class1) + [1] * len(test_class2)

    return np.array(train_data), np.array(train_labels), np.array(test_data), np.array(test_labels)
