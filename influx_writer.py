"""
InfluxDB async writer module
Handles all time-series data persistence for the FMS assessment system.
"""
from influxdb_client.client.influxdb_client_async import InfluxDBClientAsync
from influxdb_client import Point

# ── Connection config ──────────────────────────────────────────────────────────
INFLUX_URL    = "http://localhost:8086"
INFLUX_TOKEN  = "movivo"
INFLUX_ORG    = "movivo"
INFLUX_BUCKET = "movivo"


async def write_squat_data(squat_count: int, max_angle: float) -> None:
    """
    Write a single squat session data point to InfluxDB.
    Designed to be called as a fire-and-forget asyncio task.

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
                .field("squat_count", int(squat_count))
                .field("max_angle", float(max_angle))
            )
            write_api = client.write_api()
            await write_api.write(bucket=INFLUX_BUCKET, record=point)
    except Exception as e:
        print(f"[InfluxDB] write error: {e}")
