import cv2
from mediapipe import Image, ImageFormat


def change_image_format(pose_landmarker, frame):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to MediaPipe Image format
    mp_image = Image(image_format=ImageFormat.SRGB, data=image_rgb)
        
        # Detect pose landmarks
    pose_results = pose_landmarker.detect(mp_image)

        # Convert back to BGR for display
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    h, w, _ = image.shape
    return pose_results,image,h,w

def draw_skeleton(pose_results, image, h, w):
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
    for i, landmark in enumerate(landmarks):
        if i < 11 or (17 <= i <= 22):
            continue
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
    return landmarks

def draw_skeleton_right_side(pose_results, image, h, w):
    landmarks = pose_results.pose_landmarks[0]
    
    # Draw skeleton connections - Right side only
    connections = [
        (12, 14), (14, 16),  # Right arm
        (24, 26), (26, 28),  # Right leg
        (11, 12),            # Shoulders
        (23, 24),            # Hips
        (12, 24)             # Right torso
    ]
    
    for start, end in connections:
        start_pos = landmarks[start]
        end_pos = landmarks[end]
        
        start_coords = (int(start_pos.x * w), int(start_pos.y * h))
        end_coords = (int(end_pos.x * w), int(end_pos.y * h))
        
        cv2.line(image, start_coords, end_coords, (0, 255, 0), 2)
    
    # Draw landmarks as circles - Right side only
    right_landmarks = [12, 14, 16, 24, 26, 28]
    for i in right_landmarks:
        landmark = landmarks[i]
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
    
    return landmarks

def draw_skeleton_left_side(pose_results, image, h, w):
    landmarks = pose_results.pose_landmarks[0]
    
    # Draw skeleton connections - Left side only
    connections = [
        (11, 13), (13, 15),  # Left arm
        (23, 25), (25, 27),  # Left leg
        (11, 12),            # Shoulders
        (23, 24),            # Hips
        (11, 23)             # Left torso
    ]
    
    for start, end in connections:
        start_pos = landmarks[start]
        end_pos = landmarks[end]
        
        start_coords = (int(start_pos.x * w), int(start_pos.y * h))
        end_coords = (int(end_pos.x * w), int(end_pos.y * h))
        
        cv2.line(image, start_coords, end_coords, (0, 255, 0), 2)
    
    # Draw landmarks as circles - Left side only
    left_landmarks = [11, 13, 15, 23, 25, 27]
    for i in left_landmarks:
        landmark = landmarks[i]
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
    
    return landmarks