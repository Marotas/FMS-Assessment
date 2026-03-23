import numpy as np
import cv2

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

def display_angle(landmarks, point1_idx, point2_idx, point3_idx, image, w, h, color=(255, 255, 255)):
    """
    Calculate angle and display it on the image.
    
    Args:
        landmarks: The pose landmarks
        point1_idx, point2_idx, point3_idx: Landmark indices for angle calculation
        image: The image to draw on
        w, h: Image dimensions
        color: Text color (BGR format)
    
    Returns:
        The calculated angle
    """
    point1 = [landmarks[point1_idx].x * w, landmarks[point1_idx].y * h]
    point2 = [landmarks[point2_idx].x * w, landmarks[point2_idx].y * h]
    point3 = [landmarks[point3_idx].x * w, landmarks[point3_idx].y * h]
    
    angle = calculate_angle(point1, point2, point3)
    cv2.putText(image, str(int(angle)),
                (int(point2[0]), int(point2[1])),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
    
    return angle


