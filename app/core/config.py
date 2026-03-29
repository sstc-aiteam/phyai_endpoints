from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "PhyAI Endpoints"
    API_V1_STR: str = "/api/v1"
    ROBOT_IP: str = "192.168.50.75"

    # Distance from gripper center to camera along the camera's forward axis
    GRIPPER_CAMERA_OFFSET_IN_METERS: float = 0.1

    # Distance of gripper offset
    GRIPPER_OFFSET_IN_METERS: float = 0.03

    class Config:
        case_sensitive = True

settings = Settings()
