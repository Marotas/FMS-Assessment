"""PyScript application for FMS Assessment video feed with pose detection"""
import cv2
import base64
import js
import asyncio
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import utils

# Initialize the pose landmarker
def initialize_pose_landmarker():
    base_options = python.BaseOptions(model_asset_path='pose_landmarker_full.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5
    )
    return vision.PoseLandmarker.create_from_options(options)

def process_frame(pose_landmarker, frame):
    """Process frame with pose detection and draw assessment data"""
    frame = cv2.flip(frame, 1)
    pose_results, image, h, w = utils.change_image_format(pose_landmarker, frame)
    
    if pose_results.pose_landmarks:
        landmarks = utils.draw_skeleton_left_side(pose_results, image, h, w)
        utils.display_angle(landmarks, 23, 25, 27, image, w, h, (0, 255, 255))
        utils.display_angle(landmarks, 11, 23, 25, image, w, h, (0, 255, 255))
        utils.display_ankle_angle(landmarks, 25, 27, 31, image, w, h, (0, 255, 255))
    
    return image

pose_landmarker = initialize_pose_landmarker()

async def run_video_feed():
    """Run the video feed with pose assessment and display in HTML"""
    cap = cv2.VideoCapture(0)
    img_element = js.document.getElementById("videoFrame")
    status = js.document.getElementById("status")
    
    if not cap.isOpened():
        status.textContent = "Error: Could not open camera"
        return
    
    status.textContent = "Video feed active - Pose assessment running"
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame with pose detection and drawing
            image = process_frame(pose_landmarker, frame)
            
            # Convert BGR to RGB for display
            frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Encode frame to base64
            _, buffer = cv2.imencode('.jpg', frame_rgb)
            frame_base64 = base64.b64encode(buffer).decode()
            
            # Update image element
            img_element.src = f"data:image/jpeg;base64,{frame_base64}"
            img_element.style.display = "block"
            
            # Yield control to browser (~30 FPS)
            await asyncio.sleep(0.03)
    
    finally:
        cap.release()
        pose_landmarker.close()
        status.textContent = "Camera closed"

# Start the video feed
asyncio.create_task(run_video_feed())
