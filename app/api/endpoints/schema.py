from pydantic import BaseModel

class CameraCaptureResponse(BaseModel):
    image_format: str
    image_base64: str
    depth_format: str
    depth_base64: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "image_format": "png",
                    "image_base64": "iVBORw0KGgoAAAANSUhEUgAAAAE...", 
                    "depth_format": "png",
                    "depth_base64": "iVBORw0KGgoAAAANSUhEUgAAAAE..."
                }
            ]
        }
    }