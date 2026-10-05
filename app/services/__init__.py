from pydantic import BaseModel, Field


class SpeedEstimateRequest(BaseModel):
    distance_m: float = Field(..., gt=0, description="Distance traveled by the vehicle in meters")
    time_seconds: float = Field(..., gt=0, description="Observation time between checkpoints in seconds")
    vehicle_length_m: float | None = Field(default=None, gt=0, description="Optional vehicle length for calibration")
