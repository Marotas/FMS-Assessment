"""
Shared pose assessment processing module
Used by both desktop (main.py) and web (app.py) interfaces
Supports both sagittal (side view) and frontal (front view) assessment
"""
import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import utils
from utils.assessment_sagittal import SquatTracker as SagittalTracker
from utils.assessment_frontal import FrontalSquatTracker
from utils.calculations import (
    get_angle, get_trunk_lean, get_heel_lift_percentage, display_frontal_metrics
)


# Global tracker instances
sagittal_tracker = None
frontal_tracker = None
current_view_mode = "sagittal"  # Default to sagittal


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


def set_view_mode(view_mode: str):
    """
    Set the tracking view mode

    Args:
        view_mode: Either "sagittal" (side view) or "frontal" (front view)
    """
    global current_view_mode
    if view_mode.lower() not in ["sagittal", "frontal"]:
        raise ValueError(f"Invalid view mode: {view_mode}. Must be 'sagittal' or 'frontal'")
    current_view_mode = view_mode.lower()


def get_view_mode() -> str:
    """Get current view mode"""
    return current_view_mode


def get_sagittal_tracker():
    """Get or create the global sagittal tracker"""
    global sagittal_tracker
    if sagittal_tracker is None:
        sagittal_tracker = SagittalTracker()
    return sagittal_tracker


def get_frontal_tracker():
    """Get or create the global frontal tracker"""
    global frontal_tracker
    if frontal_tracker is None:
        frontal_tracker = FrontalSquatTracker()
    return frontal_tracker


def get_squat_tracker():
    """Get or create the global squat tracker"""
    global squat_tracker
    if squat_tracker is None:
        squat_tracker = utils.SquatTracker()
    return squat_tracker

def reset_all_trackers():
    """Reset all trackers"""
    global sagittal_tracker, frontal_tracker
    if sagittal_tracker is not None:
        sagittal_tracker.reset()
    if frontal_tracker is not None:
        frontal_tracker.reset()


def process_frame_sagittal(pose_landmarker, frame):
    """
    Process frame for sagittal (side view) assessment
    
    Args:
        pose_landmarker: Initialized pose landmarker
        frame: Input frame from video capture
    
    Returns:
        Tuple of (processed_image, squat_count, display_value)
    """
    # Flip the frame horizontally for a selfie-view
    frame = cv2.flip(frame, 1)
    
    # Process frame with pose detection
    pose_results, image, h, w = utils.change_image_format(pose_landmarker, frame)
    
    # Get tracker instance
    tracker = get_sagittal_tracker()
    


    # Draw pose landmarks and angles
    if pose_results.pose_landmarks:
        # Draw skeleton (left side)
        landmarks = utils.draw_skeleton_left_side(pose_results, image, h, w)
        
        # Calculate all angles for tracking (left side: shoulder 11, hip 23, knee 25, ankle 27)
        knee_angle = get_angle(landmarks, 23, 25, 27, w, h)  # hip-knee-ankle
        hip_angle = get_angle(landmarks, 11, 23, 25, w, h)   # shoulder-hip-knee
        trunk_lean_angle = get_trunk_lean(landmarks, 11, 23, w, h)  # shoulder-hip
        ankle_angle = get_angle(landmarks, 25, 27, 31, w, h)  # knee-ankle-foot
        heel_lift_percent = get_heel_lift_percentage(landmarks, 25, 27, 29, w, h)  # knee-ankle-heel
        
        # Update squat tracker with all angles and heel lift
        tracker.update(
            knee_angle=knee_angle,
            trunk_lean_angle=trunk_lean_angle,
            ankle_angle=ankle_angle,
            hip_angle=hip_angle,
            heel_lift_percent=heel_lift_percent
        )
        
        # Display all angles on image
        # Left leg angle: hip(23), knee(25), ankle(27)
        utils.display_angle(landmarks, 23, 25, 27, image, w, h, (0, 255, 255))
        
        # Left hip angle: shoulder(11), hip(23), knee(25)
        utils.display_angle(landmarks, 11, 23, 25, image, w, h, (0, 255, 255))
        
        # Left ankle angle: knee(25), ankle(27), foot(31)
        utils.display_ankle_angle(landmarks, 25, 27, 31, image, w, h, (0, 255, 255))

        # Left trunk lean: shoulder(11), hip(23)
        utils.display_trunk_lean(landmarks, 11, 23, image, w, h, (0, 255, 255))

    # Get current tracker values
    squat_count = tracker.get_count()

    # Get current rep extremes if in progress, otherwise last completed rep
    current_extremes = tracker.get_current_rep_extremes()
    if current_extremes:
        display_value = int(current_extremes['min_knee'])
    else:
        last_rep = tracker.get_rep(squat_count) if squat_count > 0 else None
        display_value = int(last_rep.min_knee_angle) if last_rep else 0

    return image, squat_count, display_value


def process_frame_frontal(pose_landmarker, frame):
    """
    Process frame for frontal (front view) assessment

    Args:
        pose_landmarker: Initialized pose landmarker
        frame: Input frame from video capture

    Returns:
        Tuple of (processed_image, squat_count, display_value)
    """
    # Flip the frame horizontally for a selfie-view
    frame = cv2.flip(frame, 1)

    # Process frame with pose detection
    pose_results, image, h, w = utils.change_image_format(pose_landmarker, frame)
    
    # Get tracker instance
    tracker = get_frontal_tracker()
    
    # Draw pose landmarks and angles
    if pose_results.pose_landmarks:
        # Draw skeleton (frontal view - both sides, no arms)
        landmarks = utils.draw_skeleton_frontal(pose_results, image, h, w)

        # Calculate frontal metrics (displays on image internally)
        left_knee, right_knee, symmetry, centering, left_align, right_align = display_frontal_metrics(
            landmarks, image, w, h, (0, 255, 255)
        )

        # Update squat tracker with frontal metrics
        tracker.update(
            left_knee_angle=left_knee,
            right_knee_angle=right_knee,
            knee_symmetry=symmetry,
            trunk_centering=centering,
            left_alignment=left_align,
            right_alignment=right_align
        )

    # Get current tracker values
    squat_count = tracker.get_count()

    # Get current rep extremes if in progress, otherwise last completed rep
    current_extremes = tracker.get_current_rep_extremes()
    if current_extremes:
        display_value = int(current_extremes['avg_symmetry'])
    else:
        last_rep = tracker.get_rep(squat_count) if squat_count > 0 else None
        display_value = int(last_rep.knee_symmetry_score) if last_rep else 0

    return image, squat_count, display_value


def process_frame(pose_landmarker, frame):
    """
    Process a frame with pose detection - dispatches to appropriate view mode

    Args:
        pose_landmarker: Initialized pose landmarker
        frame: Input frame from video capture

    Returns:
        Tuple of (processed_image, squat_count, display_value)
    """
    if current_view_mode == "frontal":
        return process_frame_frontal(pose_landmarker, frame)
    else:
        return process_frame_sagittal(pose_landmarker, frame)

