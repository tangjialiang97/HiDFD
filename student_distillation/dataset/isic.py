import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import cv2

def read_csv(path):
    return pd.read_csv(path)

def categorize_data(csv, read_dir, save_dir):
    img_id = csv['image_id']
    img_cls = csv['dx']
    for i in range(len(img_id)):
        clss = img_cls[i]
        id = img_id[i]
        sub_dir = os.path.join(save_dir, clss)
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)
        img_read_path = os.path.join(read_dir, id + '.jpg')
        img_save_path = os.path.join(sub_dir, id + '.jpg')
        img = cv2.imread(img_read_path)
        # img = cv2.resize(img, (256, 256))

        cv2.imwrite(img_save_path, img)


def merge(dir, dir1, dir2):
    if not os.path.exists(dir):
        os.makedirs(dir)
    for file in os.listdir(dir1):
        img_path = os.path.join(dir1, file)
        img = cv2.imread(img_path)
        img_save_path = os.path.join(dir, file)
        cv2.imwrite(img_save_path, img)
    for file in os.listdir(dir2):
        img_path = os.path.join(dir2, file)
        img = cv2.imread(img_path)
        img_save_path = os.path.join(dir, file)
        cv2.imwrite(img_save_path, img)

