from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.core.config import settings
from app.api.router import api_router
from app.services.realsense import realsense_service

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
