import cv2
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
from CollectFaceData import capture_face


def predict_faces(prediction_model, name):
    cam = cv2.VideoCapture(0)

    model = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
    offset = 20
    # Read image from camera

    while True:
        success, img = cam.read()
        if not success:
            print("Cannot Read From Camera")

        faces = model.detectMultiScale(img, 1.3, 5)
        i = 0
        for f in faces:
            i += 1
            x, y, w, h = f

            cropped_face = img[y - offset : y + h + offset, x - offset : x + w + offset]
            cropped_shape = cropped_face.shape
            # print(cropped_shape)
            if cropped_shape[0] > 100 and cropped_shape[1] > 100:
                cropped_face = cv2.resize(cropped_face, (100, 100))
                cropped_face = cropped_face.flatten().reshape(1, -1)
                # print(cropped_face.shape)
                # Predict class
                output = int(prediction_model.predict(cropped_face))
                # print(output)
                namePredicted = name[output]

                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    img,
                    namePredicted,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    2,
                    cv2.LINE_AA,
                )

            # cv2.imshow("Cropped" + str(i), cropped_face)

        cv2.imshow("Image Window", img)

        key = cv2.waitKey(10)
        if key == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()


choice = input("Do you want to add a new face (y/n) : ")
if choice.capitalize() == "Y":
    print(choice)
    capture_face()


# data Prepration
dataset_path = "./data/"
facedata = []
lables = []
name = {}
i = 0

for f in os.listdir(dataset_path):
    if f.endswith(".npy"):
        dataItem = np.load(dataset_path + f)
        print(dataItem.shape)
        m = dataItem.shape[0]
        target = i * np.ones((m,))
        name[i] = f[:-4]
        facedata.append(dataItem)
        lables.append(target)
        i += 1


# print(lables)
xt = np.concatenate(facedata, axis=0)
yt = np.concatenate(lables, axis=0)

print(xt.shape)
print(yt.shape)
print(name)

model = KNeighborsClassifier(n_neighbors=3)
print(model.fit(xt, yt))
predict_faces(model, name)
