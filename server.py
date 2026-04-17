import asyncio
import json
import base64
import cv2
import numpy as np
import uvicorn
import os
import uuid
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pose_processor import initialize_pose_landmarker, process_frame
from influx_writer import write_squat_data

# I used fastAPI because of th websocket that allows the server to run HTML responses
# https://fastapi.tiangolo.com/advanced/websockets/#in-production
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
os.makedirs("videos", exist_ok=True)
app.mount("/videos", StaticFiles(directory="videos"), name="videos")

pose_landmarker = initialize_pose_landmarker()

@app.get("/")
async def get():
    with open("index.html", "r", encoding="utf-8") as f:
        html = f.read()
    return HTMLResponse(html)

class AssessmentSession:
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self.is_recording = False
        self.patient_name = "Unknown"
        self.patient_id = ""
        self.session_id = ""
        self.video_writer = None
        self.video_filename = ""
        self.video_url = ""
        
        # Stats tracking for session-end persistence
        self.latest_squat_count = 0
        self.latest_max_angle = 0.0

    async def handle_command(self, data: str):
        """Processes JSON commands like start/stop recording."""
        try:
            cmd_data = json.loads(data)
            cmd = cmd_data.get("command")
            
            if cmd == "start_recording":
                self.is_recording = True
                self.patient_name = cmd_data.get("name", "Unknown")
                self.session_id = str(uuid.uuid4())
                self.patient_id = str(uuid.uuid5(uuid.NAMESPACE_OID, self.patient_name))
                
                # Sanitize filename (replace spaces with underscores)
                safe_name = self.patient_name.replace(" ", "_")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                # Switched to .webm for better browser compatibility
                self.video_filename = f"{safe_name}_{self.session_id[:8]}_{timestamp}.webm"
                self.video_url = f"{self.video_filename}"
                
                print(f"Started recording for {self.patient_name} (SessionID: {self.session_id})")
                print(f"Predetermined Video URL: {self.video_url}")
                
            elif cmd == "stop_recording":
                self.is_recording = False
                if self.video_writer:
                    self.video_writer.release()
                    self.video_writer = None
                    
                    # PERSIST TO INFLUXDB ONLY AT THE END OF SESSION
                    # We ensure the video is fully closed before saving the URL to the DB
                    print(f"Finalizing session data for InfluxDB...")
                    asyncio.create_task(write_squat_data(
                        self.latest_squat_count, 
                        self.latest_max_angle, 
                        self.patient_name, 
                        self.patient_id, 
                        self.session_id, 
                        self.video_url
                    ))
                    
                print(f"Stopped recording for {self.patient_name}")
                
        except Exception as e:
            print(f"Error parsing command: {e}")

    async def process_frame(self, data: str):
        """Decodes, processes, encodes, and records a single image frame."""
        # The client sends the image via Data URL format
        base64_data = data.split(",")[1] if data.startswith("data:image") else data
            
        try:
            # Decode base64 
            img_data = base64.b64decode(base64_data)
            nparr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                await self.websocket.send_text("error: invalid frame decoding")
                return

            # Pose Processing
            processed_img, squat_count, max_angle, knee_angle_l, knee_angle_r = process_frame(pose_landmarker, frame)
            
            # Encode response
            _, buffer = cv2.imencode('.jpg', processed_img, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
            response_b64 = base64.b64encode(buffer).decode("utf-8")
            
            response_data = {
                "image": f"data:image/jpeg;base64,{response_b64}",
                "kf_l": round(knee_angle_l, 1) if knee_angle_l is not None else None,
                "kf_r": round(knee_angle_r, 1) if knee_angle_r is not None else None
            }
            await self.websocket.send_text(json.dumps(response_data))
            
            # Save to video and InfluxDB
            # Save to class variables and video storage
            if self.is_recording:
                self.latest_squat_count = squat_count
                self.latest_max_angle = max_angle
                self._record_to_storage(processed_img, squat_count, max_angle)
                
        except Exception as e:
            print(f"Error processing frame: {e}")

    def _record_to_storage(self, img, squat_count, max_angle):
        """Internal helper for video writing."""
        if self.video_writer is None:
            os.makedirs("videos", exist_ok=True)
            h, w = img.shape[:2]
            filepath = os.path.join("videos", self.video_filename)
            
            # Switched to WebM/VP8 for browser compatibility on Linux
            print(f"Opening VideoWriter with 'VP80' (WebM) codec...")
            fourcc = cv2.VideoWriter_fourcc(*'VP80')
            self.video_writer = cv2.VideoWriter(filepath, fourcc, 30.0, (w, h))

            if not self.video_writer.isOpened():
                print(f"CRITICAL: Failed to initialize VideoWriter with 'VP80' for {filepath}")
                self.video_writer = None
                return
            else:
                print(f"VideoWriter successfully initialized.")

        self.video_writer.write(img)

    def cleanup(self):
        """Ensures resources are released."""
        if self.video_writer:
            self.video_writer.release()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session = AssessmentSession(websocket)
    
    try:
        while True:
            data = await websocket.receive_text()
            
            if data.startswith("{"):
                await session.handle_command(data)
            else:
                await session.process_frame(data)
                
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        session.cleanup()


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
