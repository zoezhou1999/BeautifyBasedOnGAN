from sklearn.svm import SVR
import numpy as np
import pickle
from generate_features import generateSymmetryFeatures,generateDistanceFeatures
from gabor_features import gabor_filter
from edge_density import canny_edge_density
from landmark_features import predictLandMarks
import cv2
import dlib
from sklearn import decomposition
import imutils
from imutils import face_utils
from sklearn.preprocessing import MinMaxScaler

image_path = 'test.jpg'
model_path = 'model.sav'
image = cv2.imread(image_path)

detector = dlib.get_frontal_face_detector()
image = imutils.resize(image, width=1920, height=1200)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect faces in the grayscale image
rects = detector(gray, 1)
with open('features.txt') as f:
    features = f.readlines()
features = [x.strip() for x in features]
X =[]
y = []
for item in features:
    mylist = item.split(',')
    X.append(list(map(float, mylist)))
cnt = 500
# loop over the face detections
for (i, rect) in enumerate(rects):
    (x, y, w, h) = face_utils.rect_to_bb(rect)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    face = image[y:y + h, x:x + w]

    f = []
    landmarkCords = predictLandMarks(image)

    ratios = generateSymmetryFeatures(landmarkCords)
    distances = generateDistanceFeatures(landmarkCords)
    gabor_features = gabor_filter(face)
    edge_density_feature = canny_edge_density(face)

    f.extend(ratios)
    f.extend(distances)
    f.extend(gabor_features)
    f.append(edge_density_feature)
    X.append(f)

    pca = decomposition.PCA(n_components=100)
    pca.fit(X)
    X_train = pca.transform(X)
    #print(X_train)
    scaler = MinMaxScaler()
    X_transform = scaler.fit_transform(X_train)
    #final_features.append(features)
    model = pickle.load(open(model_path, 'rb'))
    #print(X_train)
    res = (model.predict(X_transform))
    print(res)
    #print(res)
    disp = str("%.3f" % res[cnt])
    # show the face number
    cv2.putText(image, disp, (x - 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
    cnt += 1

cv2.imwrite("out.jpg", image)
cv2.waitKey(0)