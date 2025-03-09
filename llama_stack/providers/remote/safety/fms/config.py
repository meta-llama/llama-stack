from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union
from urllib.parse import urlparse

from llama_stack.schema_utils import json_schema_type


def resolve_detector_config(
    data: Dict[str, Any], detector_id: str
) -> Union[ContentDetectorConfig, ChatDetectorConfig]:
    """Resolve detector configuration from dictionary."""
    if isinstance(data, (ContentDetectorConfig, ChatDetectorConfig)):
        return data

    # Use the detector key as the detector_id
    data["detector_id"] = detector_id

    # Convert detector_params if present
    if "detector_params" in data and isinstance(data["detector_params"], dict):
        params = data["detector_params"]
        data["detector_params"] = DetectorParams(
            detectors=params.get("detectors"),
            **{k: v for k, v in params.items() if k != "detectors"},
        )

    # Determine detector type
    detector_type = data.pop("type", None)
    is_chat = detector_type == "chat" if detector_type else data.pop("is_chat", False)

    return ChatDetectorConfig(**data) if is_chat else ContentDetectorConfig(**data)


class MessageType(Enum):
    """Valid message types for detectors"""

    USER = "user"
    SYSTEM = "system"
    TOOL = "tool"
    COMPLETION = "completion"

    @classmethod
    def as_set(cls) -> Set[str]:
        """Get all valid message types as a set"""
        return {member.value for member in cls}


class EndpointType(Enum):
    """API endpoint types and their paths"""

    DIRECT_CONTENT = {
        "path": "/api/v1/text/contents",
        "version": "v1",
        "type": "content",
    }
    DIRECT_CHAT = {"path": "/api/v1/text/chat", "version": "v1", "type": "chat"}
    ORCHESTRATOR_CONTENT = {
        "path": "/api/v2/text/detection/content",
        "version": "v2",
        "type": "content",
    }
    ORCHESTRATOR_CHAT = {
        "path": "/api/v2/text/detection/chat",
        "version": "v2",
        "type": "chat",
    }

    @property
    def path(self) -> str:
        """Get endpoint path"""
        return self.value["path"]

    @property
    def version(self) -> str:
        """Get API version"""
        return self.value["version"]

    @property
    def type(self) -> str:
        """Get endpoint type"""
        return self.value["type"]

    @classmethod
    def get_endpoint(cls, is_orchestrator: bool, is_chat: bool) -> "EndpointType":
        """Get the appropriate endpoint type based on configuration"""
        if is_orchestrator:
            return cls.ORCHESTRATOR_CHAT if is_chat else cls.ORCHESTRATOR_CONTENT
        return cls.DIRECT_CHAT if is_chat else cls.DIRECT_CONTENT


@json_schema_type
@dataclass
class DetectorParams:
    """Common detector parameters"""

    regex: Optional[List[str]] = None
    temperature: Optional[float] = None
    risk_name: Optional[str] = None
    risk_definition: Optional[str] = None
    detectors: Optional[Dict[str, Dict[str, Any]]] = None

    def validate(self) -> None:
        """Validate detector parameters"""
        if self.temperature is not None and not 0 <= self.temperature <= 1:
            raise ValueError("Temperature must be between 0 and 1")


@json_schema_type
@dataclass
class BaseDetectorConfig:
    """Base configuration for all detectors"""

    detector_id: str
    is_chat: bool = False
    base_url: Optional[str] = None
    orchestrator_base_url: Optional[str] = None
    confidence_threshold: float = 0.5
    use_orchestrator_api: bool = False
    detector_params: Optional[DetectorParams] = None
    message_types: Set[str] = field(default_factory=lambda: MessageType.as_set())
    auth_token: Optional[str] = None

    def _validate_urls(self) -> None:
        """Validate URL configurations"""
        if not self.use_orchestrator_api and not self.base_url:
            raise ValueError("base_url is required when use_orchestrator_api is False")
        if self.use_orchestrator_api and not self.orchestrator_base_url:
            raise ValueError(
                "orchestrator_base_url is required when use_orchestrator_api is True"
            )

        for url in [self.base_url, self.orchestrator_base_url]:
            if url:
                parsed = urlparse(url)
                if not all([parsed.scheme, parsed.netloc]):
                    raise ValueError(f"Invalid URL format: {url}")
                if parsed.scheme not in {"http", "https"}:
                    raise ValueError(f"URL must use http or https scheme: {url}")

    def _validate_message_types(self) -> None:
        """Validate message type configuration"""
        if isinstance(self.message_types, (list, tuple)):
            self.message_types = set(self.message_types)

        invalid_types = self.message_types - MessageType.as_set()
        if invalid_types:
            raise ValueError(
                f"Invalid message types: {invalid_types}. "
                f"Valid types are: {MessageType.as_set()}"
            )

    def validate(self) -> None:
        """Validate configuration after all settings are propagated"""
        self._validate_message_types()
        self._validate_urls()
        if self.detector_params:
            self.detector_params.validate()

    def __post_init__(self) -> None:
        """Validate configuration immediately after initialization"""
        self._validate_message_types()

    @property
    def endpoint_type(self) -> EndpointType:
        """Get endpoint type based on configuration"""
        return EndpointType.get_endpoint(self.use_orchestrator_api, self.is_chat)


@json_schema_type
@dataclass
class ContentDetectorConfig(BaseDetectorConfig):
    """Configuration for content detectors"""

    allow_list: Optional[List[str]] = None
    block_list: Optional[List[str]] = None

    def __post_init__(self):
        self.is_chat = False
        super().__post_init__()


@json_schema_type
@dataclass
class ChatDetectorConfig(BaseDetectorConfig):
    """Configuration for chat detectors"""

    def __post_init__(self):
        self.is_chat = True
        super().__post_init__()


@json_schema_type
@dataclass
class FMSSafetyProviderConfig:
    """Configuration for the FMS Safety Provider organized by shields"""

    shields: Dict[str, Union[ContentDetectorConfig, ChatDetectorConfig]]
    orchestrator_base_url: Optional[str] = None
    use_orchestrator_api: bool = False

    def __post_init__(self):
        """Convert shield configurations to proper config objects and validate"""
        if isinstance(self.shields, dict):
            converted_shields = {}
            for shield_id, shield_config in self.shields.items():
                if isinstance(shield_config, dict):
                    converted_shields[shield_id] = resolve_detector_config(
                        shield_config, shield_id
                    )
                else:
                    converted_shields[shield_id] = shield_config
            self.shields = converted_shields

        # Run validations
        self.validate_mixed_api_usage()
        self.validate_orchestrator_config()
        self.propagate_orchestrator_settings()

        # Validate all shields
        for shield in self.all_detectors.values():
            shield.validate()

    def get_detectors_by_type(
        self, message_type: Union[str, MessageType]
    ) -> Dict[str, Union[ContentDetectorConfig, ChatDetectorConfig]]:
        """Get shields configured for a specific message type"""
        type_value = (
            message_type.value
            if isinstance(message_type, MessageType)
            else message_type
        )
        return {
            shield_id: shield
            for shield_id, shield in self.shields.items()
            if type_value in shield.message_types
        }

    @property
    def all_detectors(
        self,
    ) -> Dict[str, Union[ContentDetectorConfig, ChatDetectorConfig]]:
        """Get all shields"""
        return self.shields

    @property
    def user_message_detectors(
        self,
    ) -> Dict[str, Union[ContentDetectorConfig, ChatDetectorConfig]]:
        """Get shields configured for user messages"""
        return self.get_detectors_by_type(MessageType.USER)

    @property
    def system_message_detectors(
        self,
    ) -> Dict[str, Union[ContentDetectorConfig, ChatDetectorConfig]]:
        """Get shields configured for system messages"""
        return self.get_detectors_by_type(MessageType.SYSTEM)

    @property
    def tool_response_detectors(
        self,
    ) -> Dict[str, Union[ContentDetectorConfig, ChatDetectorConfig]]:
        """Get shields configured for tool responses"""
        return self.get_detectors_by_type(MessageType.TOOL)

    @property
    def completion_message_detectors(
        self,
    ) -> Dict[str, Union[ContentDetectorConfig, ChatDetectorConfig]]:
        """Get shields configured for completion messages"""
        return self.get_detectors_by_type(MessageType.COMPLETION)

    def _update_detector_settings(self, shield: BaseDetectorConfig) -> None:
        """Update shield settings with orchestrator configuration"""
        shield.use_orchestrator_api = True
        shield.orchestrator_base_url = self.orchestrator_base_url

    def propagate_orchestrator_settings(self) -> None:
        """Propagate orchestrator settings to all shields"""
        if self.use_orchestrator_api:
            for shield in self.all_detectors.values():
                self._update_detector_settings(shield)

    def validate_mixed_api_usage(self):
        """Check for mixed API usage across all shield types"""
        mixed_api_shields = {
            shield_id: shield.use_orchestrator_api
            for shield_id, shield in self.shields.items()
        }

        orchestrator_shields = [
            s_id for s_id, uses_orch in mixed_api_shields.items() if uses_orch
        ]
        direct_shields = [
            s_id for s_id, uses_orch in mixed_api_shields.items() if not uses_orch
        ]

        if orchestrator_shields and direct_shields:
            raise ValueError(
                "Mixed API usage detected. All shields must use either direct or orchestrator API:\n"
                f"- Orchestrator API shields: {orchestrator_shields}\n"
                f"- Direct API shields: {direct_shields}\n"
                "Please configure all shields consistently."
            )

    def validate_orchestrator_config(self):
        """Validate orchestrator configuration"""
        if self.use_orchestrator_api:
            if not self.orchestrator_base_url:
                raise ValueError(
                    "orchestrator_base_url is required when use_orchestrator_api is True"
                )

            invalid_shields = [
                shield_id
                for shield_id, shield in self.shields.items()
                if shield.base_url is not None
            ]

            if invalid_shields:
                raise ValueError(
                    f"When using orchestrator API, base_url should not be specified for shields: {invalid_shields}. "
                    "All requests will be routed through the orchestrator_base_url."
                )
