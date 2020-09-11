import cv2
import func
import os

# listup human dataset
dataset_path = '/home/cedric/Downloads/19136_796646_bundle_archive/lfw_funneled/'
list_human = []
for r, d, f in os.walk(dataset_path):
    for file in f:
        if file.endswith(".jpg") | file.endswith(".jpeg") | file.endswith(".png"):
            list_human.append(os.path.join(r, file))

# listup cat dataset
dataset_path = '/home/cedric/Downloads/13371_18106_bundle_archive/'
list_cat = []
for r, d, f in os.walk(dataset_path):
    for file in f:
        if file.endswith(".jpg") | file.endswith(".jpeg") | file.endswith(".png"):
            list_cat.append(os.path.join(r, file))

# Load the cascades
catface_cascades = cv2.CascadeClassifier('pretrained_classifier/catface.xml')
humanface_cascades = cv2.CascadeClassifier('pretrained_classifier/humanface.xml')

# human
for path in list_human:
    img = cv2.imread(path)
    human_faces = humanface_cascades.detectMultiScale(img, scaleFactor=1.3, minNeighbors=5, minSize=(75, 75))

    for (i, (x, y, w, h)) in enumerate(human_faces):
        temp = img[y:y + h, x:x + w]
        cv2.imwrite("dataset/human/{}".format(os.path.basename(path)), temp)

# cat
for path in list_cat:
    img = cv2.imread(path)
    cat_faces = catface_cascades.detectMultiScale(img, scaleFactor=1.3, minNeighbors=5, minSize=(75, 75))

    for (i, (x, y, w, h)) in enumerate(cat_faces):
        temp = img[y:y + h, x:x + w]
        cv2.imwrite("dataset/cat/{}".format(os.path.basename(path)), temp)

