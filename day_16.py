# Problem: Real-time face detection in a webcam feed using OpenCV

import cv2


# Step 1: Load the pre-trained detector Haar Cascade for face detection.
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Step 2: Capture video from teh webcam.
cap = cv2.VideoCapture(0)

# Step 3: Process each frame from the webcam feed.
while True:
    ret, frame = cap.read() # Read each frame.
    if not ret:
        break

    # Convert the image into grayscale as Haar Cascade performs better with less complex gray scale images.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Step 4: Detect the faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around the detected faces.
    for(x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the frame with detected faces.
    cv2.imshow('Face Detection', frame)

    # Press 'q' to exit the webcam.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Step 5: Release the capture and close the openCV window.
cap.release()
cv2.destroyAllWindows()