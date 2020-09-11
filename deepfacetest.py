from deepface import DeepFace
import os
import cv2
import random
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

ROOT_DIR = os.path.abspath('')
DATASET_HUMAN = os.path.join(ROOT_DIR, 'dataset', 'human')
DATASET_CAT = os.path.join(ROOT_DIR, 'dataset', 'cat')

dirs = os.listdir(DATASET_HUMAN)
files = os.listdir(DATASET_CAT)

def verify(img1_path, img2_path):
    result = DeepFace.verify(img1_path, img2_path)

# random pick*
select_data = random.randint(0, len(dirs)-1)
img1_path = os.path.join(DATASET_HUMAN, dirs[select_data])

list_val_files = []
for file in files:
    img2_path = os.path.join(DATASET_CAT, file)
    try:
        result = verify(img1_path, img2_path)
        list_val_files.append(img2_path)
    except:
        print('error')

print('done')
print(list_val_files)
