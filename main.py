import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

from utils.video_settings import change_image_format, draw_skeleton
from utils.assessment import display_angle

# Initialize the pose landmarker

base_options = python.BaseOptions(model_asset_path='pose_landmarker_full.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5
)
pose_landmarker = vision.PoseLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
cv2.namedWindow('Movivo Squat Assessment', cv2.WINDOW_NORMAL)



try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        pose_results, image, h, w = change_image_format(pose_landmarker, frame)

        # Draw pose landmarks
        if pose_results.pose_landmarks:
            landmarks = draw_skeleton(pose_results, image, h, w)
            
            # Right leg: hip(24), knee(26), ankle(28)
            display_angle(landmarks, 24, 26, 28, image, w, h, (0, 255, 255))
            
            # Left leg: hip(23), knee(25), ankle(27)
            display_angle(landmarks, 23, 25, 27, image, w, h, (0, 255, 255))

            # Right hip angle: shoulder(12), hip(24), knee(26)
            display_angle(landmarks, 12, 24, 26, image, w, h, (0, 255, 255))

            # Left hip angle: shoulder(11), hip(23), knee(25)
            display_angle(landmarks, 11, 23, 25, image, w, h, (0, 255, 255))

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