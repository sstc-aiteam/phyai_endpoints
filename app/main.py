import logging
import sys

from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.core.config import settings
from app.api.router import api_router
from app.services.realsense import realsense_service
 
# --- Logging Configuration ---
# set the log level for the root logger to INFO
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S' # Format time to seconds
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Context manager for FastAPI app lifespan events.
    Handles startup and shutdown of the RealSense service.
    """
    # Any startup logic would go here before the yield.
    yield
    # Shutdown logic is executed after the yield.
    realsense_service.shutdown()

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan
)

# Include core API router under the API_V1 prefix
app.include_router(api_router, prefix=settings.API_V1_STR)

@app.get("/")
def root():
    return {"message": f"Welcome to {settings.PROJECT_NAME}. Visit /docs for API documentation."}
