import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import Image, ImageFormat
import numpy as np

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
cv2.namedWindow('Mediapipe Feed', cv2.WINDOW_NORMAL)

def calculate_angle(a, b, c):
    """Calculate the angle between three points (shoulder, elbow, wrist)."""
    a = np.array(a)  # Shoulder
    b = np.array(b)  # Elbow
    c = np.array(c)  # Wrist

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to MediaPipe Image format
        mp_image = Image(image_format=ImageFormat.SRGB, data=image_rgb)
        
        # Detect pose landmarks
        pose_results = pose_landmarker.detect(mp_image)

        # Convert back to BGR for display
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        h, w, _ = image.shape

        # Draw pose landmarks
        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks[0]
            
            # Draw skeleton connections
            connections = [
                (11, 13), (13, 15),  # Left arm
                (12, 14), (14, 16),  # Right arm
                (11, 12),            # Shoulders
                (23, 25), (25, 27),  # Left leg
                (24, 26), (26, 28),  # Right leg
                (23, 24),            # Hips
                (11, 23), (12, 24)   # Torso
            ]
            
            for start, end in connections:
                start_pos = landmarks[start]
                end_pos = landmarks[end]
                
                start_coords = (int(start_pos.x * w), int(start_pos.y * h))
                end_coords = (int(end_pos.x * w), int(end_pos.y * h))
                
                cv2.line(image, start_coords, end_coords, (0, 255, 0), 2)
            
            # Draw landmarks as circles
            for landmark in landmarks:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
            
            # Calculate and display angles
            # Right arm: shoulder(12), elbow(14), wrist(16)
            shoulder_r = [landmarks[12].x * w, landmarks[12].y * h]
            elbow_r = [landmarks[14].x * w, landmarks[14].y * h]
            wrist_r = [landmarks[16].x * w, landmarks[16].y * h]

            angle_r = calculate_angle(shoulder_r, elbow_r, wrist_r)
            cv2.putText(image, str(int(angle_r)),
                        (int(elbow_r[0]), int(elbow_r[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Left arm: shoulder(11), elbow(13), wrist(15)
            shoulder_l = [landmarks[11].x * w, landmarks[11].y * h]
            elbow_l = [landmarks[13].x * w, landmarks[13].y * h]
            wrist_l = [landmarks[15].x * w, landmarks[15].y * h]

            angle_l = calculate_angle(shoulder_l, elbow_l, wrist_l)
            cv2.putText(image, str(int(angle_l)),
                        (int(elbow_l[0]), int(elbow_l[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Right leg: hip(24), knee(26), ankle(28)
            hip_r = [landmarks[24].x * w, landmarks[24].y * h]
            knee_r = [landmarks[26].x * w, landmarks[26].y * h]
            ankle_r = [landmarks[28].x * w, landmarks[28].y * h]

            angle_knee_r = calculate_angle(hip_r, knee_r, ankle_r)
            cv2.putText(image, str(int(angle_knee_r)),
                        (int(knee_r[0]), int(knee_r[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

            # Left leg: hip(23), knee(25), ankle(27)
            hip_l = [landmarks[23].x * w, landmarks[23].y * h]
            knee_l = [landmarks[25].x * w, landmarks[25].y * h]
            ankle_l = [landmarks[27].x * w, landmarks[27].y * h]

            angle_knee_l = calculate_angle(hip_l, knee_l, ankle_l)
            cv2.putText(image, str(int(angle_knee_l)),
                        (int(knee_l[0]), int(knee_l[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

        # Dynamically get the window size
        try:
            _, _, win_w, win_h = cv2.getWindowImageRect('Mediapipe Feed')
            image_resized = cv2.resize(image, (win_w, win_h))
        except:
            image_resized = image

        cv2.imshow('Mediapipe Feed', image_resized)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    pose_landmarker.close()