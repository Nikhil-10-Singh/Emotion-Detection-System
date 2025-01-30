
import cv2
import numpy as np
from keras.models import load_model
from keras_preprocessing.image import img_to_array

# Load the trained model
model = load_model("trained_emotion_model.h5")

# Define emotion labels
emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

# Load Haar Cascade for face detection
face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale for face detection
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.3, 5)

    for (x, y, w, h) in faces_detected:
        # Draw a rectangle around each face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), thickness=2)

        # Preprocess the face region for emotion prediction
        roi_gray = gray_img[y:y + h, x:x + w]  # Region of interest in grayscale
        roi_rgb = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2RGB)  # Convert to RGB
        roi_rgb = cv2.resize(roi_rgb, (224, 224))  # Resize to match model input
        img_pixels = img_to_array(roi_rgb)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels = img_pixels / 255.0  # Normalize the image

        # Predict emotion
        predictions = model.predict(img_pixels)
        max_index = np.argmax(predictions[0])
        predicted_emotion = emotions[max_index]

        # Display the emotion on the frame
        cv2.putText(frame, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the frame
    cv2.imshow('Facial Emotion Detection', frame)

    # Press 'q' to quit
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
