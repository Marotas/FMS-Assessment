"""
InfluxDB async writer module
Handles all time-series data persistence for the FMS assessment system.
Records frame-by-frame squat metrics for detailed analysis.
"""
from influxdb_client.client.influxdb_client_async import InfluxDBClientAsync
from influxdb_client import Point
from typing import Optional

# ── Connection config ──────────────────────────────────────────────────────────
INFLUX_URL    = "http://localhost:8086"
INFLUX_TOKEN  = "movivo"
INFLUX_ORG    = "movivo"
INFLUX_BUCKET = "movivo"


async def write_squat_data(squat_count: int, max_angle: float, patient_name: str, patient_id: str, session_id: str,
                           video_url: str = "") -> None:


    async def write_sagittal_frame(rep_number: int, frame_number: int,
                                   knee_angle: float, trunk_lean_angle: float,
                                   ankle_angle: float, hip_angle: float,
                                   heel_lift_percent: float) -> None:
        """
    Write a single frame of sagittal (side view) squat data to InfluxDB.

    Args:
        rep_number: Which squat repetition this frame belongs to
        frame_number: Sequential frame counter within the rep
        knee_angle: Current knee angle in degrees
        trunk_lean_angle: Current trunk lean in degrees
        ankle_angle: Current ankle angle in degrees
        hip_angle: Current hip angle in degrees
        heel_lift_percent: Current heel lift as percentage of shin length
    """
        try:
            async with InfluxDBClientAsync(
                    url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG
            ) as client:
                point = (
                    Point("squat_sagittal")
                    .tag("view_mode", "sagittal")
                    .tag("rep_number", int(rep_number))
                    .field("frame_number", int(frame_number))
                    .field("knee_angle", float(knee_angle))
                    .field("trunk_lean_angle", float(trunk_lean_angle))
                    .field("ankle_angle", float(ankle_angle))
                    .field("hip_angle", float(hip_angle))
                    .field("heel_lift_percent", float(heel_lift_percent))
                )
                write_api = client.write_api()
                await write_api.write(bucket=INFLUX_BUCKET, record=point)
        except Exception as e:
            print(f"[InfluxDB] sagittal write error: {e}")


async def write_frontal_frame(rep_number: int, frame_number: int,
                              left_knee_angle: float, right_knee_angle: float,
                              knee_symmetry: float, trunk_centering: float,
                              left_alignment: float, right_alignment: float) -> None:
    """
    Write a single frame of frontal (front view) squat data to InfluxDB.

    Args:
        rep_number: Which squat repetition this frame belongs to
        frame_number: Sequential frame counter within the rep
        left_knee_angle: Current left knee angle in degrees
        right_knee_angle: Current right knee angle in degrees
        knee_symmetry: Knee symmetry score (0-100%)
        trunk_centering: Trunk centering score (0-100%)
        left_alignment: Left knee-heel alignment (0-100%)
        right_alignment: Right knee-heel alignment (0-100%)
    """
    try:
        async with InfluxDBClientAsync(
                url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG
        ) as client:
            point = (
                Point("squat_frontal")
                .tag("view_mode", "frontal")
                .tag("rep_number", int(rep_number))
                .field("frame_number", int(frame_number))
                .field("left_knee_angle", float(left_knee_angle))
                .field("right_knee_angle", float(right_knee_angle))
                .field("knee_symmetry", float(knee_symmetry))
                .field("trunk_centering", float(trunk_centering))
                .field("left_alignment", float(left_alignment))
                .field("right_alignment", float(right_alignment))
            )
            write_api = client.write_api()
            await write_api.write(bucket=INFLUX_BUCKET, record=point)
    except Exception as e:
        print(f"[InfluxDB] frontal write error: {e}")


async def write_squat_data(squat_count: int, max_angle: float, patient_name: str, patient_id: str, session_id: str,
                           video_url: str = "") -> None:
    """
    Write a single squat session data point to InfluxDB.
    Designed to be called as a fire-and-forget asyncio task.

    DEPRECATED: Use write_sagittal_frame() or write_frontal_frame() instead.
    Kept for backward compatibility.

    Args:
        squat_count: Current cumulative squat count for the session
        max_angle:   Deepest knee angle recorded so far (degrees)
    """
    try:
        async with InfluxDBClientAsync(
                url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG
        ) as client:
            point = (
                Point("squat_session")
                .tag("source", "web_stream")
                .tag("patient_name", patient_name)
                .tag("patient_id", str(patient_id))
                .tag("session_id", str(session_id))
                .field("squat_count", int(squat_count))
                .field("max_angle", float(max_angle))
                .field("video_url", video_url)
            )
            write_api = client.write_api()
            await write_api.write(bucket=INFLUX_BUCKET, record=point)
    except Exception as e:
        print(f"[InfluxDB] write error: {e}")



