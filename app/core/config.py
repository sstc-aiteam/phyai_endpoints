from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "PhyAI Endpoints"
    API_V1_STR: str = "/api/v1"
    ROBOT_IP: str = "192.168.50.75"

    # Distance from TCP (wrist3) offset for the gripper length along the camera's forward axis
    GRIPPER_LEN_OFFSET_IN_METERS: float = 0.15

    # Distance to stay away from the object when approaching
    APPROACH_OFFSET_IN_METERS: float = 0.15  

    class Config:
        case_sensitive = True

settings = Settings()
