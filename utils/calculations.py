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

def get_angle(landmarks, point1_idx, point2_idx, point3_idx, w, h):
    """
    Calculate angle from landmarks without displaying it.
    
    Args:
        landmarks: The pose landmarks
        point1_idx, point2_idx, point3_idx: Landmark indices for angle calculation
        w, h: Image dimensions
    
    Returns:
        The calculated angle in degrees
    """
    point1 = [landmarks[point1_idx].x * w, landmarks[point1_idx].y * h]
    point2 = [landmarks[point2_idx].x * w, landmarks[point2_idx].y * h]
    point3 = [landmarks[point3_idx].x * w, landmarks[point3_idx].y * h]
    
    return calculate_angle(point1, point2, point3)

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

def calculate_ankle_angle(a, b, c):
    """
    Calculate the ankle angle with 0 degrees at straight position (knee, ankle, foot).
    
    Args:
        a: Knee position
        b: Ankle position (vertex of angle)
        c: Foot/Toe position
    
    Returns:
        Angle in degrees, where 0 = straight and positive values = flexion
    """
    a = np.array(a)  # Knee
    b = np.array(b)  # Ankle
    c = np.array(c)  # Foot

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    # Subtract from 180 to make 0 degrees at straight position
    return 110 - np.degrees(angle)

def display_ankle_angle(landmarks, knee_idx, ankle_idx, foot_idx, image, w, h, color=(255, 255, 255)):
    """
    Calculate ankle angle and display it on the image.
    
    Args:
        landmarks: The pose landmarks
        knee_idx, ankle_idx, foot_idx: Landmark indices for ankle angle calculation
        image: The image to draw on
        w, h: Image dimensions
        color: Text color (BGR format)
    
    Returns:
        The calculated ankle angle
    """
    knee = [landmarks[knee_idx].x * w, landmarks[knee_idx].y * h]
    ankle = [landmarks[ankle_idx].x * w, landmarks[ankle_idx].y * h]
    foot = [landmarks[foot_idx].x * w, landmarks[foot_idx].y * h]
    
    angle = calculate_ankle_angle(knee, ankle, foot)
    cv2.putText(image, str(int(angle)),
                (int(ankle[0]), int(ankle[1])),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
    
    return angle

def calculate_trunk_lean(shoulder, hip):
    """
    Calculate trunk lean angle relative to vertical (upright position = 0 degrees)
    
    Uses atan2 for proper signed angle calculation.
    Positive angles = forward lean, Negative angles = backward lean
    
    Args:
        shoulder: Shoulder position [x, y]
        hip: Hip position [x, y]
    
    Returns:
        Trunk lean angle in degrees relative to vertical
    """
    shoulder = np.array(shoulder)
    hip = np.array(hip)
    
    # Vector from hip to shoulder (along trunk)
    trunk_vector = shoulder - hip
    
    # In image coordinates: y increases downward
    # Upright trunk points upward (negative y)
    # Frame is horizontally flipped, so negate x-component to get correct direction
    # Use atan2 to get angle from vertical: atan2(-x_component, -y_component)
    # This gives positive angles for forward lean, negative for backward lean
    angle_rad = np.arctan2(-trunk_vector[0], -trunk_vector[1])
    trunk_lean = np.degrees(angle_rad)
    
    return trunk_lean

def get_trunk_lean(landmarks, shoulder_idx, hip_idx, w, h):
    """
    Calculate trunk lean from landmarks without displaying it.
    
    Args:
        landmarks: The pose landmarks
        shoulder_idx, hip_idx: Landmark indices for trunk lean calculation
        w, h: Image dimensions
    
    Returns:
        The calculated trunk lean angle in degrees
    """
    shoulder = [landmarks[shoulder_idx].x * w, landmarks[shoulder_idx].y * h]
    hip = [landmarks[hip_idx].x * w, landmarks[hip_idx].y * h]
    
    return calculate_trunk_lean(shoulder, hip)

def display_trunk_lean(landmarks, shoulder_idx, hip_idx, image, w, h, color=(255, 255, 255)):
    """
    Calculate trunk lean angle and display it on the image.
    
    Args:
        landmarks: The pose landmarks
        shoulder_idx, hip_idx: Landmark indices for trunk lean calculation
        image: The image to draw on
        w, h: Image dimensions
        color: Text color (BGR format)
    
    Returns:
        The calculated trunk lean angle
    """
    shoulder = [landmarks[shoulder_idx].x * w, landmarks[shoulder_idx].y * h]
    hip = [landmarks[hip_idx].x * w, landmarks[hip_idx].y * h]
    
    angle = calculate_trunk_lean(shoulder, hip)
    
    # Display at the midpoint between shoulder and hip
    midpoint = (int((shoulder[0] + hip[0]) / 2), int((shoulder[1] + hip[1]) / 2))
    cv2.putText(image, str(int(angle)),
                midpoint,
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
    
    return angle

def get_shin_length(landmarks, knee_idx, ankle_idx, w, h):
    """
    Calculate shin segment length (knee to ankle distance).
    
    Used for normalizing heel lift measurements.
    
    Args:
        landmarks: The pose landmarks
        knee_idx, ankle_idx: Landmark indices for knee and ankle
        w, h: Image dimensions
    
    Returns:
        Shin length in pixels
    """
    knee = np.array([landmarks[knee_idx].x * w, landmarks[knee_idx].y * h])
    ankle = np.array([landmarks[ankle_idx].x * w, landmarks[ankle_idx].y * h])
    
    shin_length = np.linalg.norm(ankle - knee)
    return shin_length

def calculate_heel_lift_percentage(heel_y, ankle_y, shin_length):
    """
    Calculate heel lift as a percentage of shin length.
    
    Args:
        heel_y: Y-coordinate of heel (in pixels, image coordinates where y increases downward)
        ankle_y: Y-coordinate of ankle (baseline standing position)
        shin_length: Length of shin segment in pixels (for normalization)
    
    Returns:
        Heel lift as percentage of shin length. 0% = heel on ground, higher values = heel lifted more
    """
    if shin_length == 0:
        return 0
    
    # Heel lift distance in pixels (negative y means upward in image coords)
    lift_distance = max(0, ankle_y - heel_y)
    
    # Normalize by shin length and convert to percentage
    heel_lift_percent = (lift_distance / shin_length) * 100
    
    return heel_lift_percent

def get_heel_lift_percentage(landmarks, knee_idx, ankle_idx, heel_idx, w, h):
    """
    Calculate heel lift percentage from landmarks.
    
    Args:
        landmarks: The pose landmarks
        knee_idx, ankle_idx, heel_idx: Landmark indices for knee, ankle, and heel
        w, h: Image dimensions
    
    Returns:
        Heel lift as percentage of shin length
    """
    heel_y = landmarks[heel_idx].y * h
    ankle_y = landmarks[ankle_idx].y * h
    shin_length = get_shin_length(landmarks, knee_idx, ankle_idx, w, h)
    
    return calculate_heel_lift_percentage(heel_y, ankle_y, shin_length)

def display_heel_lift(landmarks, knee_idx, ankle_idx, heel_idx, image, w, h, color=(255, 255, 255)):
    """
    Calculate heel lift and display it on the image.
    
    Args:
        landmarks: The pose landmarks
        knee_idx, ankle_idx, heel_idx: Landmark indices for knee, ankle, and heel
        image: The image to draw on
        w, h: Image dimensions
        color: Text color (BGR format)
    
    Returns:
        Heel lift as percentage of shin length
    """
    heel_lift_percent = get_heel_lift_percentage(landmarks, knee_idx, ankle_idx, heel_idx, w, h)
    
    # Display at heel position
    heel_x = landmarks[heel_idx].x * w
    heel_y = landmarks[heel_idx].y * h
    
    cv2.putText(image, f"Heel Lift: {heel_lift_percent:.1f}%",
                (int(heel_x), int(heel_y)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    
    return heel_lift_percent

def calculate_knee_heel_alignment(knee_x, heel_x, shin_length):
    """
    Calculate how well knee and heel are aligned (vertically in image).
    
    Measures if heel x-coordinate is directly under knee x-coordinate.
    
    Args:
        knee_x: X-coordinate of knee
        heel_x: X-coordinate of heel
        shin_length: Length of shin segment (for normalization)
    
    Returns:
        Alignment score 0-100%, where 100 = perfectly aligned
    """
    if shin_length == 0:
        return 0
    
    # Horizontal distance between knee and heel
    alignment_distance = abs(knee_x - heel_x)
    
    # Normalize by shin length to get percentage deviation
    deviation_percent = (alignment_distance / shin_length) * 100
    
    # Convert to alignment score: 0% deviation = 100% alignment
    alignment_score = max(0, 100 - deviation_percent)
    
    return alignment_score

def get_knee_heel_alignment(landmarks, left_knee_idx, left_heel_idx, left_ankle_idx, 
                            right_knee_idx, right_heel_idx, right_ankle_idx, w, h):
    """
    Calculate knee-heel alignment for both legs.
    
    Args:
        landmarks: The pose landmarks
        left_knee_idx, left_heel_idx, left_ankle_idx: Left side indices
        right_knee_idx, right_heel_idx, right_ankle_idx: Right side indices
        w, h: Image dimensions
    
    Returns:
        Tuple of (left_alignment_score, right_alignment_score) both 0-100%
    """
    left_knee_x = landmarks[left_knee_idx].x * w
    left_heel_x = landmarks[left_heel_idx].x * w
    left_shin_length = get_shin_length(landmarks, left_knee_idx, left_ankle_idx, w, h)
    
    right_knee_x = landmarks[right_knee_idx].x * w
    right_heel_x = landmarks[right_heel_idx].x * w
    right_shin_length = get_shin_length(landmarks, right_knee_idx, right_ankle_idx, w, h)
    
    left_alignment = calculate_knee_heel_alignment(left_knee_x, left_heel_x, left_shin_length)
    right_alignment = calculate_knee_heel_alignment(right_knee_x, right_heel_x, right_shin_length)
    
    return left_alignment, right_alignment

def calculate_trunk_centering(left_shoulder_x, right_shoulder_x, left_hip_x, right_hip_x, w):
    """
    Calculate how centered trunk is (not leaning side-to-side).
    
    Measures if shoulder midpoint and hip midpoint are aligned vertically.
    
    Args:
        left_shoulder_x, right_shoulder_x: X-coordinates of shoulders
        left_hip_x, right_hip_x: X-coordinates of hips
        w: Image width (for normalization)
    
    Returns:
        Centering score 0-100%, where 100 = perfectly centered
    """
    # Calculate midpoints
    shoulder_midpoint_x = (left_shoulder_x + right_shoulder_x) / 2
    hip_midpoint_x = (left_hip_x + right_hip_x) / 2
    
    # Distance between midpoints
    centering_distance = abs(shoulder_midpoint_x - hip_midpoint_x)
    
    # Normalize by image width
    deviation_percent = (centering_distance / w) * 100
    
    # Convert to centering score
    centering_score = max(0, 100 - deviation_percent)
    
    return centering_score

def get_trunk_centering(landmarks, left_shoulder_idx, right_shoulder_idx, 
                        left_hip_idx, right_hip_idx, w, h):
    """
    Calculate trunk centering from landmarks.
    
    Args:
        landmarks: The pose landmarks
        left_shoulder_idx, right_shoulder_idx: Shoulder landmark indices
        left_hip_idx, right_hip_idx: Hip landmark indices
        w, h: Image dimensions
    
    Returns:
        Trunk centering score 0-100%
    """
    left_shoulder_x = landmarks[left_shoulder_idx].x * w
    right_shoulder_x = landmarks[right_shoulder_idx].x * w
    left_hip_x = landmarks[left_hip_idx].x * w
    right_hip_x = landmarks[right_hip_idx].x * w
    
    return calculate_trunk_centering(left_shoulder_x, right_shoulder_x, left_hip_x, right_hip_x, w)

def calculate_knee_symmetry(left_knee_angle, right_knee_angle):
    """
    Calculate how symmetric both knees are during squat.
    
    Measures if left and right knee angles are similar.
    
    Args:
        left_knee_angle: Left knee angle in degrees
        right_knee_angle: Right knee angle in degrees
    
    Returns:
        Symmetry score 0-100%, where 100 = perfectly symmetric knees
    """
    # Calculate angle difference
    angle_difference = abs(left_knee_angle - right_knee_angle)
    
    # Convert difference to symmetry score (max 30 degree difference = 0 symmetry)
    max_acceptable_difference = 30
    symmetry_percent = max(0, 100 - (angle_difference / max_acceptable_difference * 100))
    
    return symmetry_percent

def display_frontal_metrics(landmarks, image, w, h, color=(0, 255, 255)):
    """
    Calculate and display frontal view metrics on image.
    
    Args:
        landmarks: The pose landmarks
        image: The image to draw on
        w, h: Image dimensions
        color: Text color (BGR format)
    
    Returns:
        Tuple of (left_knee_angle, right_knee_angle, symmetry_score, trunk_centering_score, left_alignment, right_alignment)
    """
    # Left leg angles: hip(23), knee(25), ankle(27) and foot(31)
    left_knee_angle = get_angle(landmarks, 23, 25, 27, w, h)
    
    # Right leg angles: hip(24), knee(26), ankle(28) and foot(32)
    right_knee_angle = get_angle(landmarks, 24, 26, 28, w, h)
    
    # Knee symmetry
    symmetry_score = calculate_knee_symmetry(left_knee_angle, right_knee_angle)
    
    # Trunk centering
    trunk_centering_score = get_trunk_centering(landmarks, 11, 12, 23, 24, w, h)
    
    # Knee-heel alignment
    left_alignment, right_alignment = get_knee_heel_alignment(
        landmarks, 25, 29, 27, 26, 30, 28, w, h
    )
    
    # Display on image
    y_start = 80
    cv2.putText(image, f"L-Knee: {int(left_knee_angle)}°  R-Knee: {int(right_knee_angle)}°",
                (10, y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1, cv2.LINE_AA)
    cv2.putText(image, f"Symmetry: {symmetry_score:.0f}%",
                (10, y_start + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1, cv2.LINE_AA)
    cv2.putText(image, f"Trunk Center: {trunk_centering_score:.0f}%",
                (10, y_start + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1, cv2.LINE_AA)
    cv2.putText(image, f"L-Align: {left_alignment:.0f}%  R-Align: {right_alignment:.0f}%",
                (10, y_start + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1, cv2.LINE_AA)
    
    return left_knee_angle, right_knee_angle, symmetry_score, trunk_centering_score, left_alignment, right_alignment

