"""
Shared pose assessment processing module
Used by both desktop (main.py) and web (app.py) interfaces
"""
import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import utils


def initialize_pose_landmarker():
    """Initialize and return the pose landmarker"""
    base_options = python.BaseOptions(model_asset_path='pose_landmarker_full.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5
    )
    return vision.PoseLandmarker.create_from_options(options)


def process_frame(pose_landmarker, frame):
    """
    Process a frame with pose detection and draw assessment data
    
    Args:
        pose_landmarker: Initialized pose landmarker
        frame: Input frame from video capture
    
    Returns:
        Processed image with skeleton and angles drawn
    """
    # Flip the frame horizontally for a selfie-view
    frame = cv2.flip(frame, 1)
    
    # Process frame with pose detection
    pose_results, image, h, w = utils.change_image_format(pose_landmarker, frame)
    
    # Draw pose landmarks and angles
    if pose_results.pose_landmarks:
        # Draw skeleton (left side)
        landmarks = utils.draw_skeleton_left_side(pose_results, image, h, w)
        
        # Left leg angle: hip(23), knee(25), ankle(27)
        utils.display_angle(landmarks, 23, 25, 27, image, w, h, (0, 255, 255))
        
        # Left hip angle: shoulder(11), hip(23), knee(25)
        utils.display_angle(landmarks, 11, 23, 25, image, w, h, (0, 255, 255))
        
        # Left ankle angle: knee(25), ankle(27), foot(31)
        utils.display_ankle_angle(landmarks, 25, 27, 31, image, w, h, (0, 255, 255))
    
    return image
