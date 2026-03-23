import cv2
import numpy as np

from pose_processor import initialize_pose_landmarker, process_frame

# Initialize the pose landmarker
pose_landmarker = initialize_pose_landmarker()

cap = cv2.VideoCapture(1)
cv2.namedWindow('Movivo Squat Assessment', cv2.WINDOW_NORMAL)

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame with pose detection and drawing
        image = process_frame(pose_landmarker, frame)

        # Dynamically get the window size
        try:
            _, _, win_w, win_h = cv2.getWindowImageRect('Mediapipe Feed')
            image_resized = cv2.resize(image, (win_w, win_h))
        except:
            image_resized = image

        cv2.imshow('Movivo Squat Assessment', image_resized)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    pose_landmarker.close()