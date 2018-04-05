import cv2

import numpy as np
import keras
model = keras.models.load_model('gender.model')

# capture video from webcam
cap = cv2.VideoCapture(0)
# initialize face detection
cascPath = 'haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)

while(True):
    # read a frame. frame is in BGR( Blue, Green, Red) format
    ret, frame = cap.read()

    # convert the image to grayscale for performing face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # create a rgb frame since our model expects its input image
    # to be in RGB format
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    

    cropped_faces = []
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        # increase the width, height of face region by some pixels.
        # we want a bit more pixels rather than just the face
        extra_pad = 40
        x = max(1, x - extra_pad)
        y = max(1, y - extra_pad)
        width = min(frame.shape[1], x + w + extra_pad*2)
        height = min(frame.shape[0], y + h + extra_pad*2)
        cv2.rectangle(rgb_frame, (x, y), (width, height), (0, 255, 255), 2)

        # crop the face
        cropped = rgb_frame[y: height, x: width].astype("float32")
        cropped = cv2.resize(cropped, (198, 198))
        
        # need to rescale the values from 0 to 1
        cropped = cropped /255.0
        cropped = np.clip(cropped, 0, 1)
        
        cropped_faces.append(cropped)


    # predict the faces
    cv2.putText(rgb_frame, "Total faces = {}".format(len(faces)), (10, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (30, 255, 200), 2)
    if len(faces) > 0 and 1 == 1:
        predictions = model.predict(np.array(cropped_faces))
        predictions = np.where(predictions.flatten() < 0.5, "female", "male")
        for i, prediction in enumerate(predictions):
            print(prediction)
            text = prediction

            x, y, _, _ = faces[i]
            cv2.putText(rgb_frame, text, (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    
    # convert our RGB frame back to BGR format before displaying
    cv2.imshow('frame', cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()    