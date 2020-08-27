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

'''
# imread

img1 = cv2.imread('human.jpeg')
img2 = cv2.imread('cat.jpeg')

# cascades
human_faces = humanface_cascades.detectMultiScale(img1, scaleFactor=1.3, minNeighbors=5, minSize=(75, 75))
cat_faces = catface_cascades.detectMultiScale(img2, scaleFactor=1.3, minNeighbors=5, minSize=(75, 75))

for (i, (x, y, w, h)) in enumerate(human_faces):
    temp = img1[y:y+h, x:x+w]
    cv2.imwrite("only_face.png", temp)
for (i, (x, y, w, h)) in enumerate(cat_faces):
    temp = img2[y:y+h, x:x+w]
    cv2.imwrite("only_cat_face.png", temp)
#dst = cv2.resize(src, dsize=(640, 480), interpolation=cv2.INTER_AREA)
#cv2.cvtColor(original
#img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
#print(glob.glob("/home/cedric/Downloads/19136_796646_bundle_archive/lfw_funneled/*.jpg"))
'''

