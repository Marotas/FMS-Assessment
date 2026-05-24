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
from pydantic import BaseModel
from pose_processor import (
    initialize_pose_landmarker, process_frame, set_view_mode, get_view_mode,
    get_last_sagittal_metrics, get_last_frontal_metrics, reset_all_trackers
)
from influx_writer import write_sagittal_frame, write_frontal_frame, write_session_summary

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

active_sessions = {}
connected_clients = []

class StartRecordingRequest(BaseModel):
    patient_id: str
    patient_name: str
    view_mode: str

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
                self.session_id = cmd_data.get("session_id", str(uuid.uuid4()))
                self.patient_id = cmd_data.get("patient_id", str(uuid.uuid5(uuid.NAMESPACE_OID, self.patient_name)))
                
                view_mode = cmd_data.get("view_mode", "sagittal")
                set_view_mode(view_mode)
                reset_all_trackers()
                
                active_sessions[self.patient_id] = self

                # Sanitize filename (replace spaces with underscores)
                safe_name = self.patient_name.replace(" ", "_")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                # Switched to .webm for better browser compatibility
                self.video_filename = f"{safe_name}_{self.session_id[:8]}_{timestamp}.webm"
                self.video_url = f"{self.video_filename}"

                print(f"Started recording for {self.patient_name} (SessionID: {self.session_id})")
                print(f"Predetermined Video URL: {self.video_url}")

            elif cmd == "stop_recording":
                self.finalize_session()
                print(f"Stopped recording for {self.patient_name}")

        except Exception as e:
            print(f"Error parsing command: {e}")

    def finalize_session(self):
        """Finalizes the recording session, saves video and influxDB summary."""
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None

        if self.is_recording:
            print(f"Finalizing session data for {self.patient_name}...")
            fms_score = 2 if self.latest_squat_count > 0 else 0
            view_mode = get_view_mode()
            asyncio.create_task(write_session_summary(
                self.latest_squat_count,
                self.latest_max_angle,
                fms_score,
                self.patient_name,
                self.patient_id,
                self.session_id,
                self.video_url,
                view_mode
            ))
            self.is_recording = False
            if self.patient_id in active_sessions:
                del active_sessions[self.patient_id]

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

            # Pose Processing (returns 3 values now)
            processed_img, squat_count, max_angle = process_frame(pose_landmarker, frame)

            # Encode response
            _, buffer = cv2.imencode('.jpg', processed_img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            response_b64 = base64.b64encode(buffer).decode("utf-8")

            # Fetch metrics and build JSON payload
            view_mode = get_view_mode()
            payload = {
                "image": f"data:image/jpeg;base64,{response_b64}",
                "view_mode": view_mode
            }

            if view_mode == "sagittal":
                metrics = get_last_sagittal_metrics()
                if metrics:
                    payload.update({
                        "kf_l": metrics.get('knee_angle'),
                        "hf_l": metrics.get('hip_angle'),
                        "df_l": metrics.get('ankle_angle'),
                        "tv_val": metrics.get('trunk_lean_angle'),
                        "fh_l": metrics.get('heel_lift_percent')
                    })
            elif view_mode == "frontal":
                metrics = get_last_frontal_metrics()
                if metrics:
                    payload.update({
                        "kf_l": metrics.get('left_knee_angle'),
                        "kf_r": metrics.get('right_knee_angle'),
                        "as_val": metrics.get('knee_symmetry'),
                        "ts_val": metrics.get('trunk_centering')
                    })

            await self.websocket.send_text(json.dumps(payload))

            # Save to video and InfluxDB
            if self.is_recording:
                self.latest_squat_count = squat_count
                self.latest_max_angle = max_angle
                self._record_to_storage(processed_img, squat_count, max_angle)

                # Fire-and-forget: write detailed metrics to InfluxDB based on view mode
                view_mode = get_view_mode()

                if view_mode == "sagittal":
                    metrics = get_last_sagittal_metrics()
                    if metrics:
                        asyncio.create_task(write_sagittal_frame(
                            rep_number=metrics['rep_number'],
                            frame_number=metrics['frame_number'],
                            knee_angle=metrics['knee_angle'],
                            trunk_lean_angle=metrics['trunk_lean_angle'],
                            ankle_angle=metrics['ankle_angle'],
                            hip_angle=metrics['hip_angle'],
                            heel_lift_percent=metrics['heel_lift_percent'],
                            patient_name=self.patient_name,
                            patient_id=self.patient_id,
                            session_id=self.session_id
                        ))
                elif view_mode == "frontal":
                    metrics = get_last_frontal_metrics()
                    if metrics:
                        asyncio.create_task(write_frontal_frame(
                            rep_number=metrics['rep_number'],
                            frame_number=metrics['frame_number'],
                            left_knee_angle=metrics['left_knee_angle'],
                            right_knee_angle=metrics['right_knee_angle'],
                            knee_symmetry=metrics['knee_symmetry'],
                            trunk_centering=metrics['trunk_centering'],
                            left_alignment=metrics['left_alignment'],
                            right_alignment=metrics['right_alignment'],
                            patient_name=self.patient_name,
                            patient_id=self.patient_id,
                            session_id=self.session_id
                        ))

        except Exception as e:
            print(f"Error processing frame: {e}")

    def _record_to_storage(self, img, squat_count, max_angle):
        """Internal helper for video writing."""
        if self.video_writer is None:
            os.makedirs("videos", exist_ok=True)
            h, w = img.shape[:2]

            # Try codecs in order of preference
            codecs = [
                ('VP80', '.webm'),
                ('mp4v', '.mp4'),
                ('XVID', '.avi'),
            ]

            for codec, ext in codecs:
                base_name = os.path.splitext(self.video_filename)[0]
                test_filename = f"{base_name}{ext}"
                filepath = os.path.join("videos", test_filename)

                print(f"Trying VideoWriter with '{codec}' codec...")
                fourcc = cv2.VideoWriter_fourcc(*codec)
                writer = cv2.VideoWriter(filepath, fourcc, 30.0, (w, h))

                if writer.isOpened():
                    self.video_writer = writer
                    self.video_filename = test_filename
                    self.video_url = test_filename
                    print(f"VideoWriter initialized with '{codec}' -> {test_filename}")
                    break
                else:
                    writer.release()
                    print(f"Codec '{codec}' not available, trying next...")

            if self.video_writer is None:
                print(f"CRITICAL: No working video codec found!")
                return

        self.video_writer.write(img)

    def cleanup(self):
        """Ensures resources are released."""
        self.finalize_session()


@app.post("/api/start_recording")
async def api_start_recording(req: StartRecordingRequest):
    if not connected_clients:
        return {"status": "error", "message": "No FMS client connected"}
    
    session = connected_clients[-1] # use the most recently connected client
    
    cmd_data = {
        "command": "start_recording",
        "name": req.patient_name,
        "patient_id": req.patient_id,
        "view_mode": req.view_mode
    }
    
    # Process it directly
    await session.handle_command(json.dumps(cmd_data))
    
    # Update UI on the FMS client
    try:
        await session.websocket.send_text(json.dumps({
            "action": "ui_update",
            "state": "RECORDING",
            "message": f"Aufnahme läuft für {req.patient_name}... ({req.view_mode})"
        }))
    except:
        pass

    return {"status": "started"}

@app.post("/api/stop_recording/{patient_id}")
async def stop_recording_api(patient_id: str):
    if patient_id in active_sessions:
        session = active_sessions[patient_id]
        session.finalize_session()
        # Notify the websocket client to update its UI
        try:
            asyncio.create_task(session.websocket.send_text(json.dumps({
                "action": "ui_update",
                "state": "IDLE",
                "message": "Aufnahme beendet. Warte auf nächste Sitzung..."
            })))
        except:
            pass
        return {"status": "stopped"}
    return {"status": "not_found"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session = AssessmentSession(websocket)
    connected_clients.append(session)
    try:
        while True:
            data = await websocket.receive_text()
            if data.startswith("{"):
                await session.handle_command(data)
            else:
                await session.process_frame(data)
    except WebSocketDisconnect:
        print("Client disconnected")
        if session in connected_clients:
            connected_clients.remove(session)
        session.cleanup()
    except Exception as e:
        print(f"WebSocket error: {e}")


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
