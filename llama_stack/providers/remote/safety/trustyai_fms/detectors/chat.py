from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, cast

from llama_stack.apis.inference import Message
from llama_stack.apis.safety import RunShieldResponse
from llama_stack.providers.remote.safety.trustyai_fms.config import ChatDetectorConfig
from llama_stack.providers.remote.safety.trustyai_fms.detectors.base import (
    BaseDetector,
    DetectionResult,
    DetectorError,
    DetectorRequestError,
    DetectorValidationError,
)

# Type aliases for better readability
ChatMessage = Dict[
    str, Any
]  # Changed from Dict[str, str] to Dict[str, Any] to handle complex content
ChatRequest = Dict[str, Any]
DetectorResponse = List[Dict[str, Any]]

logger = logging.getLogger(__name__)


class ChatDetectorError(DetectorError):
    """Specific errors for chat detector operations"""

    pass


@dataclass(frozen=True)
class ChatDetectionMetadata:
    """Structured metadata for chat detections"""

    risk_name: Optional[str] = None
    risk_definition: Optional[str] = None
    additional_metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary format"""
        result: Dict[str, Any] = {}  # Fixed the type annotation here
        if self.risk_name:
            result["risk_name"] = self.risk_name
        if self.risk_definition:
            result["risk_definition"] = self.risk_definition
        if self.additional_metadata:
            result["metadata"] = self.additional_metadata
        return result


class ChatDetector(BaseDetector):
    """Detector for chat-based safety checks"""

    def __init__(self, config: ChatDetectorConfig) -> None:
        """Initialize chat detector with configuration"""
        if not isinstance(config, ChatDetectorConfig):
            raise DetectorValidationError(
                "Config must be an instance of ChatDetectorConfig"
            )
        super().__init__(config)
        self.config: ChatDetectorConfig = config
        logger.info(f"Initialized ChatDetector with config: {vars(config)}")

    def _extract_detector_params(self) -> Dict[str, Any]:
        """Extract non-null detector parameters"""
        if not self.config.detector_params:
            return {}

        params = {
            k: v for k, v in vars(self.config.detector_params).items() if v is not None
        }
        logger.debug(f"Extracted detector params: {params}")
        return params

    def _prepare_chat_request(
        self, messages: List[ChatMessage], params: Optional[Dict[str, Any]] = None
    ) -> ChatRequest:
        """Prepare the request based on API mode"""
        # Format messages for detector API
        formatted_messages: List[Dict[str, str]] = []  # Explicitly typed
        for msg in messages:
            formatted_msg = {
                "content": str(msg.get("content", "")),  # Ensure string type
                "role": "user",  # Always send as user for detector API
            }
            formatted_messages.append(formatted_msg)

        if self.config.use_orchestrator_api:
            payload: Dict[str, Any] = {
                "messages": formatted_messages
            }  # Explicitly typed

            # Initialize detector_config to avoid None
            detector_config: Dict[str, Any] = {}  # Explicitly typed

            # NEW STRUCTURE: Check for top-level detectors first
            if hasattr(self.config, "detectors") and self.config.detectors:
                for detector_id, det_config in self.config.detectors.items():
                    detector_config[detector_id] = det_config.get("detector_params", {})

            # LEGACY STRUCTURE: Check for nested detectors
            elif (
                self.config.detector_params
                and hasattr(self.config.detector_params, "detectors")
                and self.config.detector_params.detectors
            ):
                detector_config = self.config.detector_params.detectors

            # Handle flat params - group them into generic containers
            elif self.config.detector_params:
                detector_params = self._extract_detector_params()

                # Create a flat dictionary of parameters
                flat_params = {}

                # Extract from model_params
                if "model_params" in detector_params and isinstance(
                    detector_params["model_params"], dict
                ):
                    flat_params.update(detector_params["model_params"])

                # Extract from metadata
                if "metadata" in detector_params and isinstance(
                    detector_params["metadata"], dict
                ):
                    flat_params.update(detector_params["metadata"])

                # Extract from kwargs
                if "kwargs" in detector_params and isinstance(
                    detector_params["kwargs"], dict
                ):
                    flat_params.update(detector_params["kwargs"])

                # Add any other direct parameters, but skip container dictionaries
                for k, v in detector_params.items():
                    if (
                        k not in ["model_params", "metadata", "kwargs", "params"]
                        and v is not None
                    ):
                        flat_params[k] = v

                # Add all flattened parameters directly to detector configuration
                detector_config[self.config.detector_id] = flat_params

            # Ensure we have a valid detectors map even if all checks fail
            if not detector_config:
                detector_config = {self.config.detector_id: {}}

            payload["detectors"] = detector_config
            return payload

        # Direct API format remains unchanged
        else:
            # DIRECT MODE: Use flat parameters for API compatibility
            # Don't organize into containers for direct mode
            detector_params = self._extract_detector_params()

            # Flatten the parameters for direct mode too
            flat_params = {}

            # Extract from model_params
            if "model_params" in detector_params and isinstance(
                detector_params["model_params"], dict
            ):
                flat_params.update(detector_params["model_params"])

            # Extract from metadata
            if "metadata" in detector_params and isinstance(
                detector_params["metadata"], dict
            ):
                flat_params.update(detector_params["metadata"])

            # Extract from kwargs
            if "kwargs" in detector_params and isinstance(
                detector_params["kwargs"], dict
            ):
                flat_params.update(detector_params["kwargs"])

            # Add any other direct parameters
            for k, v in detector_params.items():
                if (
                    k not in ["model_params", "metadata", "kwargs", "params"]
                    and v is not None
                ):
                    flat_params[k] = v

            return {
                "messages": formatted_messages,
                "detector_params": flat_params if flat_params else params or {},
            }

    async def _call_detector_api(
        self,
        messages: List[ChatMessage],
        params: Optional[Dict[str, Any]] = None,
    ) -> DetectorResponse:
        """Call chat detector API with proper endpoint selection"""
        try:
            request = self._prepare_chat_request(messages, params)
            headers = self._prepare_headers()

            logger.info("Making detector API request")
            logger.debug(f"Request headers: {headers}")
            logger.debug(f"Request payload: {request}")

            response = await self._make_request(request, headers)
            return self._extract_detections(response)

        except Exception as e:
            logger.error(f"API call failed: {str(e)}", exc_info=True)
            raise DetectorRequestError(
                f"Chat detector API call failed: {str(e)}"
            ) from e

    def _extract_detections(self, response: Dict[str, Any]) -> DetectorResponse:
        """Extract detections from API response"""
        if not response:
            logger.debug("Empty response received")
            return []

        if self.config.use_orchestrator_api:
            detections = response.get("detections", [])
            if not detections:
                # Add default detection when none returned
                logger.debug("No detections found, adding default low-score detection")
                return [
                    {
                        "detection_type": "risk",
                        "detection": "No",
                        "detector_id": self.config.detector_id,
                        "score": 0.0,  # Default low score
                    }
                ]
            logger.debug(f"Orchestrator detections: {detections}")
            return cast(
                DetectorResponse, detections
            )  # Explicit cast to correct return type

        # Direct API returns a list where first item contains detections
        if isinstance(response, list) and response:
            detections = (
                [response[0]] if not isinstance(response[0], list) else response[0]
            )
            logger.debug(f"Direct API detections: {detections}")
            return cast(
                DetectorResponse, detections
            )  # Explicit cast to correct return type

        logger.debug("No detections found in response")
        return []

    def _process_detection(
        self, detection: Dict[str, Any]
    ) -> Tuple[
        Optional[DetectionResult], float
    ]:  # Changed return type to match base class
        """Process detection result and validate against threshold"""
        score = detection.get("score", 0.0)  # Default to 0.0 if score is missing

        if score > self.score_threshold:
            metadata = ChatDetectionMetadata(
                risk_name=(
                    self.config.detector_params.risk_name
                    if self.config.detector_params
                    else None
                ),
                risk_definition=(
                    self.config.detector_params.risk_definition
                    if self.config.detector_params
                    else None
                ),
                additional_metadata=detection.get("metadata"),
            )

            result = DetectionResult(
                detection="Yes",
                detection_type=detection["detection_type"],
                score=score,
                detector_id=detection.get("detector_id", self.config.detector_id),
                text=detection.get("text", ""),
                start=detection.get("start", 0),
                end=detection.get("end", 0),
                metadata=metadata.to_dict(),
            )
            return (result, score)
        return (None, score)

    async def _run_shield_impl(
        self,
        shield_id: str,
        messages: List[Message],
        params: Optional[Dict[str, Any]] = None,
    ) -> RunShieldResponse:
        """Implementation of shield checks for chat messages"""
        try:
            shield = await self.shield_store.get_shield(shield_id)
            self._validate_shield(shield)

            logger.info(f"Processing {len(messages)} message(s)")

            # Convert messages keeping only necessary fields
            chat_messages: List[ChatMessage] = []  # Explicitly typed
            for msg in messages:
                message_dict: ChatMessage = {"content": msg.content, "role": msg.role}
                # Preserve type if present for internal processing
                if hasattr(msg, "type"):
                    message_dict["type"] = msg.type
                chat_messages.append(message_dict)

            logger.debug(f"Prepared messages: {chat_messages}")
            detections = await self._call_detector_api(chat_messages, params)

            for detection in detections:
                processed, score = self._process_detection(detection)
                if processed:
                    logger.info(f"Violation detected: {processed}")
                    return self.create_violation_response(
                        processed, detection.get("detector_id", self.config.detector_id)
                    )

            logger.debug("No violations detected")
            return RunShieldResponse()

        except Exception as e:
            logger.error(f"Shield execution failed: {str(e)}", exc_info=True)
            raise ChatDetectorError(f"Shield execution failed: {str(e)}") from e
