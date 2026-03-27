import asyncio
import base64
import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pose_processor import initialize_pose_landmarker, process_frame
from influx_writer import write_squat_data

# I used fastAPI because of th websocket that allows the server to run HTML responses
# https://fastapi.tiangolo.com/advanced/websockets/#in-production
app = FastAPI()

pose_landmarker = initialize_pose_landmarker()

@app.get("/")
async def get():
    with open("index.html", "r", encoding="utf-8") as f:
        html = f.read()
    return HTMLResponse(html)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Receive frame as base64 string from the client browser
            data = await websocket.receive_text()
            
            # The client sends the image via Data URL format
            if data.startswith("data:image"):
                base64_data = data.split(",")[1]
            else:
                base64_data = data
                
            try:
                # Decode base64 back into an OpenCV image
                img_data = base64.b64decode(base64_data)
                nparr = np.frombuffer(img_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is not None:
                    # Process the frame using your existing logic
                    # This will draw the skeleton, angles, and update squat counts
                    processed_img, squat_count, max_angle = process_frame(pose_landmarker, frame)
                    
                    # Encode the processed image to send back to the browser
                    _, buffer = cv2.imencode('.jpg', processed_img, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                    response_b64 = base64.b64encode(buffer).decode("utf-8")
                    
                    # Send the processed frame back to the client
                    await websocket.send_text(f"data:image/jpeg;base64,{response_b64}")
                    
                    # Fire-and-forget: write to InfluxDB without blocking the stream
                    asyncio.create_task(write_squat_data(squat_count, max_angle))
                else:
                    await websocket.send_text("error: invalid frame decoding")
            except Exception as cv_e:
                print(f"Error processing frame: {cv_e}")
                
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
