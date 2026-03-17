from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "PhyAI Endpoints"
    API_V1_STR: str = "/api/v1"
    ROBOT_IP: str = "192.168.50.75"

    class Config:
        case_sensitive = True

settings = Settings()
