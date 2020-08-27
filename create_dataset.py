import cv2
import func

# Load the cascades
catface_cascades = cv2.CascadeClassifier('catface.xml')
humanface_cascades = cv2.CascadeClassifier('humanface.xml')

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

img_1 = cv2.imread("only_cat_face.png")
img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
img_1 = cv2.resize(img_1, dsize=(100, 100), interpolation=cv2.INTER_AREA)

img_2 = cv2.imread("only_cat_face.png")
img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
img_2 = cv2.resize(img_2, dsize=(100, 100), interpolation=cv2.INTER_AREA)

test = func.mse(img_1, img_2)
print(test)

