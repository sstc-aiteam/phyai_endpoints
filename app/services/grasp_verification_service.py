import json
import logging

from openai import OpenAI, OpenAIError

from app.core.config import settings

logger = logging.getLogger(__name__)


class GraspVerificationError(Exception):
    """Raised when AI grasp verification cannot be completed."""


class GraspVerificationService:
    def __init__(self):
        self._client: OpenAI | None = None

    def _get_client(self) -> OpenAI:
        if not settings.OPENAI_API_KEY:
            raise GraspVerificationError("OPENAI_API_KEY is not configured.")

        if self._client is None:
            self._client = OpenAI(api_key=settings.OPENAI_API_KEY)
        return self._client

    def verify_grasp(self, image_base64: str, image_mime_type: str = "image/jpeg") -> dict:
        """
        Uses a vision-capable OpenAI model to determine whether the gripper is holding an object.
        """
        prompt = (
            "You are verifying a robot gripper after a grasp attempt. "
            "Look at the image and decide whether the gripper is currently holding any physical object. "
            "Return true only when an object is visibly caught or held by the gripper. "
            "Return false when the gripper is empty, the object has fallen, or the image is too unclear to confirm."
        )

        schema = {
            "type": "object",
            "properties": {
                "is_grasped": {
                    "type": "boolean",
                    "description": "True if the gripper is visibly holding an object.",
                },
                "confidence": {
                    "type": "number",
                    "description": "Confidence from 0.0 to 1.0.",
                    "minimum": 0,
                    "maximum": 1,
                },
                "reason": {
                    "type": "string",
                    "description": "Brief reason for the decision.",
                },
            },
            "required": ["is_grasped", "confidence", "reason"],
            "additionalProperties": False,
        }

        try:
            response = self._get_client().responses.create(
                model=settings.OPENAI_GRASP_VERIFICATION_MODEL,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {
                                "type": "input_image",
                                "image_url": f"data:{image_mime_type};base64,{image_base64}",
                            },
                        ],
                    }
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "grasp_verification",
                        "strict": True,
                        "schema": schema,
                    }
                },
            )
        except OpenAIError as e:
            logger.error("OpenAI grasp verification request failed: %s", e, exc_info=True)
            raise GraspVerificationError(f"OpenAI request failed: {e}") from e

        try:
            result = json.loads(response.output_text)
        except (TypeError, json.JSONDecodeError) as e:
            logger.error("Failed to parse OpenAI grasp verification response: %r", response, exc_info=True)
            raise GraspVerificationError("OpenAI response was not valid JSON.") from e

        return {
            "is_grasped": bool(result["is_grasped"]),
            "confidence": float(result["confidence"]),
            "reason": str(result["reason"]),
        }


grasp_verification_service = GraspVerificationService()
