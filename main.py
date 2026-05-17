import cv2
import numpy as np

from pose_processor import (
    initialize_pose_landmarker, process_frame, get_squat_tracker,
    set_view_mode, get_view_mode, get_sagittal_tracker, get_frontal_tracker
)


def select_view_mode():
    """Prompt user to select tracking view mode"""
    print("\n" + "="*60)
    print("MOVIVO SQUAT ASSESSMENT")
    print("="*60)
    print("\nSelect tracking view mode:")
    print("1. Sagittal (Side View) - Tracks knee angle, trunk lean, ankle, hip, heel lift")
    print("2. Frontal (Front View) - Tracks knee symmetry, trunk centering, knee-heel alignment")
    print("\nEnter choice (1 or 2): ", end="")
    
    choice = input().strip()
    if choice == "2":
        set_view_mode("frontal")
        print("✓ Frontal view mode selected")
    else:
        set_view_mode("sagittal")
        print("✓ Sagittal view mode selected")
    
    print("="*60)
    print("Press 'q' to quit assessment\n")


# Select view mode at startup
select_view_mode()

# Initialize the pose landmarker
pose_landmarker = initialize_pose_landmarker()

cap = cv2.VideoCapture(0)
cv2.namedWindow('Movivo Squat Assessment', cv2.WINDOW_NORMAL)

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame with pose detection and drawing
        image, squat_count, display_value = process_frame(pose_landmarker, frame)
        
        view_mode = get_view_mode()
        
        # Display current rep metrics on screen based on view mode
        tracker = get_squat_tracker()
        current_extremes = tracker.get_current_rep_extremes()
        
        if current_extremes:
            y_offset = 80
            cv2.putText(image, f"Current Rep:",
                        (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
            
            if view_mode == "sagittal":
                # Sagittal view metrics
                cv2.putText(image, f"  Knee: {int(current_extremes['min_knee'])}°",
                            (10, y_offset + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 1, cv2.LINE_AA)
                cv2.putText(image, f"  Trunk: {int(current_extremes['max_trunk_lean'])}°",
                            (10, y_offset + 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 1, cv2.LINE_AA)
                cv2.putText(image, f"  Ankle: {int(current_extremes['min_ankle'])}°",
                            (10, y_offset + 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 1, cv2.LINE_AA)
                cv2.putText(image, f"  Hip: {int(current_extremes['min_hip'])}°",
                            (10, y_offset + 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 1, cv2.LINE_AA)
                cv2.putText(image, f"  Heel Lift: {current_extremes['max_heel_lift']:.1f}%",
                            (10, y_offset + 125),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 1, cv2.LINE_AA)
            else:
                # Frontal view metrics
                cv2.putText(image, f"  L-Knee: {int(current_extremes['left_knee'])}°  R-Knee: {int(current_extremes['right_knee'])}°",
                            (10, y_offset + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 1, cv2.LINE_AA)
                cv2.putText(image, f"  Symmetry: {current_extremes['avg_symmetry']:.0f}%",
                            (10, y_offset + 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 1, cv2.LINE_AA)
                cv2.putText(image, f"  Trunk Center: {current_extremes['avg_trunk_centering']:.0f}%",
                            (10, y_offset + 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 1, cv2.LINE_AA)
                cv2.putText(image, f"  L-Align: {current_extremes['left_alignment']:.0f}%  R-Align: {current_extremes['right_alignment']:.0f}%",
                            (10, y_offset + 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 1, cv2.LINE_AA)

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
    
    # Print session summary
    view_mode = get_view_mode()
    
    if view_mode == "sagittal":
        tracker = get_sagittal_tracker()
    else:
        tracker = get_frontal_tracker()
    
    squat_count = tracker.get_count()
    print("\n" + "="*60)
    print(f"SESSION SUMMARY - {squat_count} squats completed ({view_mode.upper()} View)")
    print("="*60)
    
    if squat_count > 0:
        for rep in tracker.get_all_reps():
            print(rep)
    else:
        print("No completed squats in this session.")
    
    print("="*60)
