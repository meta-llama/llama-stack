from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, cast

from llama_stack.apis.inference import Message
from llama_stack.apis.safety import RunShieldResponse
from llama_stack.providers.remote.safety.trustyai_fms.config import (
    ContentDetectorConfig,
)
from llama_stack.providers.remote.safety.trustyai_fms.detectors.base import (
    BaseDetector,
    DetectionResult,
    DetectorError,
    DetectorRequestError,
    DetectorValidationError,
)

# Type aliases for better readability
ContentRequest = Dict[str, Any]
DetectorResponse = List[Dict[str, Any]]

logger = logging.getLogger(__name__)


class ContentDetectorError(DetectorError):
    """Specific errors for content detector operations"""

    pass


class ContentDetector(BaseDetector):
    """Detector for content-based safety checks"""

    def __init__(self, config: ContentDetectorConfig) -> None:
        """Initialize content detector with configuration"""
        if not isinstance(config, ContentDetectorConfig):
            raise DetectorValidationError(
                "Config must be an instance of ContentDetectorConfig"
            )
        super().__init__(config)
        self.config: ContentDetectorConfig = config
        logger.info(f"Initialized ContentDetector with config: {vars(config)}")

    def _extract_detector_params(self) -> Dict[str, Any]:
        """Extract detector parameters with support for generic format"""
        if not self.config.detector_params:
            return {}

        # Use to_dict() to flatten our categorized structure into what the API expects
        params = self.config.detector_params.to_dict()
        logger.debug(f"Extracted detector params: {params}")
        return params

    def _prepare_content_request(
        self, content: str, params: Optional[Dict[str, Any]] = None
    ) -> ContentRequest:
        """Prepare the request based on API mode"""

        if self.config.use_orchestrator_api:
            payload: Dict[str, Any] = {"content": content}  # Always use singular form

            # NEW STRUCTURE: Check for top-level detectors first
            if hasattr(self.config, "detectors") and self.config.detectors:
                detector_config: Dict[str, Any] = {}
                for detector_id, det_config in self.config.detectors.items():
                    detector_config[detector_id] = det_config.get("detector_params", {})
                payload["detectors"] = detector_config
                return payload

            # LEGACY STRUCTURE: Check for nested detectors
            elif self.config.detector_params and hasattr(
                self.config.detector_params, "detectors"
            ):
                detectors = getattr(self.config.detector_params, "detectors", {})
                payload["detectors"] = detectors
                return payload

            # Handle flat params
            else:
                detector_config = {}
                detector_params = self._extract_detector_params()
                if detector_params:
                    detector_config[self.config.detector_id] = detector_params
                payload["detectors"] = detector_config
                return payload

        else:
            # DIRECT MODE: Use flat parameters for API compatibility
            detector_params = self._extract_detector_params()

            return {
                "contents": [content],
                "detector_params": detector_params if detector_params else params or {},
            }

    def _extract_detections(self, response: Dict[str, Any]) -> DetectorResponse:
        """Extract detections from API response"""
        if not response:
            logger.debug("Empty response received")
            return []

        if self.config.use_orchestrator_api:
            detections = response.get("detections", [])
            logger.debug(f"Orchestrator detections: {detections}")
            return cast(List[Dict[str, Any]], detections)

        # Direct API returns a list of lists where inner list contains detections
        if isinstance(response, list) and response:
            detections = response[0] if isinstance(response[0], list) else [response[0]]
            logger.debug(f"Direct API detections: {detections}")
            return cast(List[Dict[str, Any]], detections)

        logger.debug("No detections found in response")
        return []

    async def _call_detector_api(
        self,
        content: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> DetectorResponse:
        """Call detector API with proper endpoint selection"""
        try:
            request = self._prepare_content_request(content, params)
            headers = self._prepare_headers()

            logger.info("Making detector API request")
            logger.debug(f"Request headers: {headers}")
            logger.debug(f"Request payload: {request}")

            response = await self._make_request(request, headers)
            return self._extract_detections(response)

        except Exception as e:
            logger.error(f"API call failed: {str(e)}", exc_info=True)
            raise DetectorRequestError(
                f"Content detector API call failed: {str(e)}"
            ) from e

    def _process_detection(
        self, detection: Dict[str, Any]
    ) -> tuple[Optional[DetectionResult], float]:
        """Process detection result and validate against threshold"""
        if not detection.get("score"):
            logger.warning("Detection missing score field")
            return None, 0.0

        score = detection.get("score", 0)
        if score > self.score_threshold:
            result = DetectionResult(
                detection="Yes",
                detection_type=detection["detection_type"],
                score=score,
                detector_id=detection.get("detector_id", self.config.detector_id),
                text=detection.get("text", ""),
                start=detection.get("start", 0),
                end=detection.get("end", 0),
                metadata=detection.get("metadata", {}),
            )
            return result, score
        return None, score

    async def _run_shield_impl(
        self,
        shield_id: str,
        messages: List[Message],
        params: Optional[Dict[str, Any]] = None,
    ) -> RunShieldResponse:
        """Implementation of shield checks for content messages"""
        try:
            shield = await self.shield_store.get_shield(shield_id)
            self._validate_shield(shield)

            for msg in messages:
                content = msg.content
                content_str: str

                # Simplified content handling - just check for text attribute or convert to string
                if hasattr(content, "text"):
                    content_str = str(content.text)
                elif isinstance(content, list):
                    content_str = " ".join(
                        str(getattr(item, "text", "")) for item in content
                    )
                else:
                    content_str = str(content)

                truncated_content = (
                    content_str[:100] + "..." if len(content_str) > 100 else content_str
                )
                logger.debug(f"Checking content: {truncated_content}")

                detections = await self._call_detector_api(content_str, params)

                for detection in detections:
                    processed, score = self._process_detection(detection)
                    if processed:
                        logger.info(f"Violation detected: {processed}")
                        return self.create_violation_response(
                            processed,
                            detection.get("detector_id", self.config.detector_id),
                        )

            logger.debug("No violations detected")
            return RunShieldResponse()

        except Exception as e:
            logger.error(f"Shield execution failed: {str(e)}", exc_info=True)
            raise ContentDetectorError(f"Shield execution failed: {str(e)}") from e
