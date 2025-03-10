from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from llama_stack.apis.inference import Message
from llama_stack.apis.safety import RunShieldResponse
from llama_stack.providers.remote.safety.fms.config import (
    ContentDetectorConfig,
)
from llama_stack.providers.remote.safety.fms.detectors.base import (
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


@dataclass(frozen=True)
class ContentDetectionMetadata:
    """Structured metadata for content detections"""

    allow_list_match: Optional[str] = None
    block_list_match: Optional[str] = None
    additional_metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary format"""
        result = {}
        if self.allow_list_match:
            result["allow_list_match"] = self.allow_list_match
        if self.block_list_match:
            result["block_list_match"] = self.block_list_match
        if self.additional_metadata:
            result["metadata"] = self.additional_metadata
        return result


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
        """Extract non-null detector parameters"""
        if not self.config.detector_params:
            return {}

        params = {
            k: v for k, v in vars(self.config.detector_params).items() if v is not None
        }
        logger.debug(f"Extracted detector params: {params}")
        return params

    def _prepare_content_request(
        self, content: str, params: Optional[Dict[str, Any]] = None
    ) -> ContentRequest:
        """Prepare the request based on API mode"""
        detector_params = self._extract_detector_params()

        if self.config.use_orchestrator_api:
            # Use detectors configuration directly from detector_params if available
            if (
                hasattr(self.config.detector_params, "detectors")
                and self.config.detector_params.detectors
            ):
                return {
                    "detectors": self.config.detector_params.detectors,
                    "content": content,
                }

        # Fallback to legacy format
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

        """Extract detections from API response"""
        if self.config.use_orchestrator_api:
            detections = response.get("detections", [])
            logger.debug(f"Orchestrator detections: {detections}")
            return detections

        # Direct API returns a list of lists where inner list contains detections
        if isinstance(response, list) and response:
            detections = response[0] if isinstance(response[0], list) else [response[0]]
            logger.debug(f"Direct API detections: {detections}")
            return detections

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
    ) -> Optional[DetectionResult]:
        """Process detection result and validate against threshold"""
        if not detection.get("score"):
            logger.warning("Detection missing score field")
            return None

        score = detection.get("score", 0)
        if score > self.score_threshold:
            metadata = ContentDetectionMetadata(
                allow_list_match=detection.get("allow_list_match"),
                block_list_match=detection.get("block_list_match"),
                additional_metadata=detection.get("metadata"),
            )

            return DetectionResult(
                detection="Yes",
                detection_type=detection["detection_type"],
                score=score,
                detector_id=detection.get("detector_id", self.config.detector_id),
                text=detection.get("text", ""),
                start=detection.get("start", 0),
                end=detection.get("end", 0),
                metadata=metadata.to_dict(),
            )
        return None

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

            contents = [msg.content for msg in messages]
            logger.info(f"Processing {len(contents)} message(s)")

            for content in contents:
                truncated_content = (
                    content[:100] + "..." if len(content) > 100 else content
                )
                logger.debug(f"Checking content: {truncated_content}")

                detections = await self._call_detector_api(content, params)

                for detection in detections:
                    if processed := self._process_detection(detection):
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
