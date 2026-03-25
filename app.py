"""PyScript application for FMS Assessment video feed with pose detection"""
import cv2
import base64
import js
import asyncio
from pose_processor import initialize_pose_landmarker, process_frame

# Initialize the pose landmarker
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
            image, squat_count, max_angle = process_frame(pose_landmarker, frame)
            
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

