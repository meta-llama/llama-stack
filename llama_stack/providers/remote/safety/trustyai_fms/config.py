from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union
from urllib.parse import urlparse

from pydantic import BaseModel, Field, model_validator

from llama_stack.schema_utils import json_schema_type


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

    @classmethod
    def get_endpoint(cls, is_orchestrator: bool, is_chat: bool) -> EndpointType:
        """Get appropriate endpoint based on configuration"""
        if is_orchestrator:
            return cls.ORCHESTRATOR_CHAT if is_chat else cls.ORCHESTRATOR_CONTENT
        return cls.DIRECT_CHAT if is_chat else cls.DIRECT_CONTENT


@json_schema_type
@dataclass
class DetectorParams:
    """Flexible parameter container supporting nested structure and arbitrary parameters"""

    # Store all parameters in a single dictionary for maximum flexibility
    params: Dict[str, Any] = field(default_factory=dict)

    # Parameter categories for organization
    model_params: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    kwargs: Dict[str, Any] = field(default_factory=dict)

    # Store detectors directly as an attribute (not in params) for orchestrator mode
    _raw_detectors: Optional[Dict[str, Dict[str, Any]]] = None

    # Standard parameters kept for backward compatibility
    @property
    def regex(self) -> Optional[List[str]]:
        return self.params.get("regex")

    @regex.setter
    def regex(self, value: List[str]) -> None:
        self.params["regex"] = value

    @property
    def temperature(self) -> Optional[float]:
        return self.model_params.get("temperature") or self.params.get("temperature")

    @temperature.setter
    def temperature(self, value: float) -> None:
        self.model_params["temperature"] = value

    @property
    def risk_name(self) -> Optional[str]:
        return self.metadata.get("risk_name") or self.params.get("risk_name")

    @risk_name.setter
    def risk_name(self, value: str) -> None:
        self.metadata["risk_name"] = value

    @property
    def risk_definition(self) -> Optional[str]:
        return self.metadata.get("risk_definition") or self.params.get(
            "risk_definition"
        )

    @risk_definition.setter
    def risk_definition(self, value: str) -> None:
        self.metadata["risk_definition"] = value

    @property
    def orchestrator_detectors(self) -> Dict[str, Dict[str, Any]]:
        """Return detectors in the format required by orchestrator API"""
        if (
            not hasattr(self, "_detectors") or not self._detectors
        ):  # Direct attribute access
            return {}

        flattened = {}
        for (
            detector_id,
            detector_config,
        ) in self._detectors.items():  # Direct attribute access
            # Create a flattened version without extra nesting
            flat_config = {}

            # Extract detector_params if present and flatten them
            params = detector_config.get("detector_params", {})
            if isinstance(params, dict):
                # Move params up to top level
                for key, value in params.items():
                    flat_config[key] = value

            flattened[detector_id] = flat_config

        return flattened

    @property
    def formatted_detectors(self) -> Optional[Dict[str, Dict[str, Any]]]:
        """Return detectors properly formatted for orchestrator API"""
        # Direct return for API usage - avoid calling other properties
        if hasattr(self, "_detectors") and self._detectors:
            return self.orchestrator_detectors
        return None

    @formatted_detectors.setter
    def formatted_detectors(self, value: Dict[str, Dict[str, Any]]) -> None:
        self._detectors = value

    @property
    def detectors(self) -> Optional[Dict[str, Dict[str, Any]]]:
        """COMPATIBILITY: Returns the same as formatted_detectors to maintain API compatibility"""
        # Using a different implementation to avoid the redefinition error
        # while maintaining the same functionality
        if not hasattr(self, "_detectors") or not self._detectors:
            return None
        return self.orchestrator_detectors

    @detectors.setter
    def detectors(self, value: Dict[str, Dict[str, Any]]) -> None:
        """COMPATIBILITY: Set detectors while maintaining compatibility"""
        self._detectors = value

    # And fix the __setitem__ method:
    def __setitem__(self, key: str, value: Any) -> None:
        """Allow dictionary-like assignment with smart categorization"""
        # Special handling for known params
        if key == "detectors":
            self._detectors = value  # Set underlying attribute directly
            return
        elif key == "regex":
            self.params[key] = value
            return

        # Rest of the method remains unchanged
        known_model_params = ["temperature", "top_p", "top_k", "max_tokens", "n"]
        known_metadata = ["risk_name", "risk_definition", "category", "severity"]

        if key in known_model_params:
            self.model_params[key] = value
        elif key in known_metadata:
            self.metadata[key] = value
        else:
            self.kwargs[key] = value

    def __init__(self, **kwargs):
        """Initialize from any keyword arguments with smart categorization"""
        # Initialize containers
        self.params = {}
        self.model_params = {}
        self.metadata = {}
        self.kwargs = {}
        self._raw_detectors = None

        # Special handling for nested detectors structure
        if "detectors" in kwargs:
            self._raw_detectors = kwargs.pop("detectors")

        # Special handling for regex
        if "regex" in kwargs:
            self.params["regex"] = kwargs.pop("regex")

        # Categorize known parameters
        known_model_params = ["temperature", "top_p", "top_k", "max_tokens", "n"]
        known_metadata = ["risk_name", "risk_definition", "category", "severity"]

        # Explicit categories if provided
        if "model_params" in kwargs:
            self.model_params.update(kwargs.pop("model_params"))

        if "metadata" in kwargs:
            self.metadata.update(kwargs.pop("metadata"))

        if "kwargs" in kwargs:
            self.kwargs.update(kwargs.pop("kwargs"))

        # Categorize remaining parameters
        for key, value in kwargs.items():
            if key in known_model_params:
                self.model_params[key] = value
            elif key in known_metadata:
                self.metadata[key] = value
            else:
                self.kwargs[key] = value

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-like access with category lookup"""
        if key in self.params:
            return self.params[key]
        elif key in self.model_params:
            return self.model_params[key]
        elif key in self.metadata:
            return self.metadata[key]
        elif key in self.kwargs:
            return self.kwargs[key]
        return None

    def get(self, key: str, default: Any = None) -> Any:
        """Dictionary-style get with category lookup"""
        result = self.__getitem__(key)
        return default if result is None else result

    def set(self, key: str, value: Any) -> None:
        """Set a parameter value with smart categorization"""
        self.__setitem__(key, value)

    def update(self, params: Dict[str, Any]) -> None:
        """Update with multiple parameters, respecting categories"""
        for key, value in params.items():
            self.__setitem__(key, value)

    def to_dict(self) -> Dict[str, Any]:
        """Convert all parameters to a flat dictionary for API requests"""
        result = {}

        # Add core parameters
        result.update(self.params)

        # Add all categorized parameters, flattened
        result.update(self.model_params)
        result.update(self.metadata)
        result.update(self.kwargs)

        return result

    def to_categorized_dict(self) -> Dict[str, Any]:
        """Convert to a structured dictionary with categories preserved"""
        result = dict(self.params)

        if self.model_params:
            result["model_params"] = dict(self.model_params)

        if self.metadata:
            result["metadata"] = dict(self.metadata)

        if self.kwargs:
            result["kwargs"] = dict(self.kwargs)

        return result

    def create_flattened_detector_configs(self) -> Dict[str, Dict[str, Any]]:
        """Create flattened detector configurations for orchestrator mode.
        This removes the extra detector_params nesting that causes API errors.
        """
        if not self.detectors:
            return {}

        flattened = {}
        for detector_id, detector_config in self.detectors.items():
            # Create a flattened version without extra nesting
            flat_config = {}

            # Extract detector_params if present and flatten them
            params = detector_config.get("detector_params", {})
            if isinstance(params, dict):
                # Move params up to top level
                for key, value in params.items():
                    flat_config[key] = value

            flattened[detector_id] = flat_config

        return flattened

    def validate(self) -> None:
        """Validate parameter values"""
        if self.temperature is not None and not 0 <= self.temperature <= 1:
            raise ValueError("Temperature must be between 0 and 1")


@json_schema_type
@dataclass
class BaseDetectorConfig:
    """Base configuration for all detectors with flexible parameter handling"""

    detector_id: str
    confidence_threshold: float = 0.5
    message_types: Set[str] = field(default_factory=lambda: MessageType.as_set())
    auth_token: Optional[str] = None
    detector_params: Optional[DetectorParams] = None

    # URL fields directly on detector configs
    detector_url: Optional[str] = None
    orchestrator_url: Optional[str] = None

    # Flexible storage for any additional parameters
    _extra_params: Dict[str, Any] = field(default_factory=dict)

    # Runtime execution parameters
    max_concurrency: int = 10  # Maximum concurrent API requests
    request_timeout: float = 30.0  # HTTP request timeout in seconds
    max_retries: int = 3  # Maximum number of retry attempts
    backoff_factor: float = 1.5  # Exponential backoff multiplier
    max_keepalive_connections: int = 5  # Max number of keepalive connections
    max_connections: int = 10  # Max number of connections in the pool

    @property
    def use_orchestrator_api(self) -> bool:
        """Determine if orchestrator API should be used"""
        return bool(self.orchestrator_url)

    def __post_init__(self) -> None:
        """Process configuration after initialization"""
        # Convert list/tuple message_types to set
        if isinstance(self.message_types, (list, tuple)):
            self.message_types = set(self.message_types)

        # Validate message types
        invalid_types = self.message_types - MessageType.as_set()
        if invalid_types:
            raise ValueError(
                f"Invalid message types: {invalid_types}. "
                f"Valid types are: {MessageType.as_set()}"
            )

        # Initialize detector_params if needed
        if self.detector_params is None:
            self.detector_params = DetectorParams()

        # Handle legacy URL field names
        if hasattr(self, "base_url") and self.base_url and not self.detector_url:
            self.detector_url = self.base_url

        if (
            hasattr(self, "orchestrator_base_url")
            and self.orchestrator_base_url
            and not self.orchestrator_url
        ):
            self.orchestrator_url = self.orchestrator_base_url

    def validate(self) -> None:
        """Validate configuration"""
        # Validate detector_params
        if self.detector_params:
            self.detector_params.validate()

        # Validate that at least one URL is provided
        if not self.detector_url and not self.orchestrator_url:
            raise ValueError(f"No URL provided for detector {self.detector_id}")

        # Validate URLs if present
        for url_name, url in [
            ("detector_url", self.detector_url),
            ("orchestrator_url", self.orchestrator_url),
        ]:
            if url:
                self._validate_url(url, url_name)

    def _validate_url(self, url: str, url_name: str) -> None:
        """Validate URL format"""
        parsed = urlparse(url)
        if not all([parsed.scheme, parsed.netloc]):
            raise ValueError(f"Invalid {url_name} format: {url}")
        if parsed.scheme not in {"http", "https"}:
            raise ValueError(f"Invalid {url_name} scheme: {parsed.scheme}")

    def get(self, key: str, default: Any = None) -> Any:
        """Get parameter with fallback to extra parameters"""
        try:
            return getattr(self, key)
        except AttributeError:
            return self._extra_params.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set parameter, storing in extra_params if not a standard field"""
        if hasattr(self, key) and key not in ["_extra_params"]:
            setattr(self, key, value)
        else:
            self._extra_params[key] = value

    @property
    def is_chat(self) -> bool:
        """Default implementation, should be overridden by subclasses"""
        return False


@json_schema_type
@dataclass
class ContentDetectorConfig(BaseDetectorConfig):
    """Configuration for content detectors"""

    @property
    def is_chat(self) -> bool:
        """Content detectors are not chat detectors"""
        return False


@json_schema_type
@dataclass
class ChatDetectorConfig(BaseDetectorConfig):
    """Configuration for chat detectors"""

    @property
    def is_chat(self) -> bool:
        """Chat detectors are chat detectors"""
        return True


@json_schema_type
class FMSSafetyProviderConfig(BaseModel):
    """Configuration for the FMS Safety Provider"""

    shields: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    # Rename _detectors to remove the leading underscore
    detectors_internal: Dict[str, Union[ContentDetectorConfig, ChatDetectorConfig]] = (
        Field(default_factory=dict, exclude=True)
    )

    # Provider-level orchestrator URL (can be copied to shields if needed)
    orchestrator_url: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True

    # Add a model validator to replace __post_init__
    @model_validator(mode="after")
    def setup_config(self):
        """Process shield configurations"""
        # Process shields into detector objects
        self._process_shields()

        # Replace shields dictionary with processed detector configs
        self.shields = self.detectors_internal

        # Validate all shields
        for shield in self.shields.values():
            shield.validate()

        return self

    def _process_shields(self):
        """Process all shield configurations into detector configs"""
        for shield_id, config in self.shields.items():
            if isinstance(config, dict):
                # Copy the config to avoid modifying the original
                shield_config = dict(config)

                # Check if this shield has nested detectors
                nested_detectors = shield_config.pop("detectors", None)

                # Determine detector type
                detector_type = shield_config.pop("type", None)
                is_chat = (
                    detector_type == "chat"
                    if detector_type
                    else shield_config.pop("is_chat", False)
                )

                # Set detector ID
                shield_config["detector_id"] = shield_id

                # Handle URL fields
                # First handle legacy field names
                if "base_url" in shield_config and "detector_url" not in shield_config:
                    shield_config["detector_url"] = shield_config.pop("base_url")

                if (
                    "orchestrator_base_url" in shield_config
                    and "orchestrator_url" not in shield_config
                ):
                    shield_config["orchestrator_url"] = shield_config.pop(
                        "orchestrator_base_url"
                    )

                # If no orchestrator_url in shield but provider has one, copy it
                if self.orchestrator_url and "orchestrator_url" not in shield_config:
                    shield_config["orchestrator_url"] = self.orchestrator_url

                # Initialize detector_params with proper structure for nested detectors
                detector_params_dict = shield_config.get("detector_params", {})
                if not isinstance(detector_params_dict, dict):
                    detector_params_dict = {}

                # Create detector_params object
                detector_params = DetectorParams(**detector_params_dict)

                # Add nested detectors if present
                if nested_detectors:
                    detector_params.detectors = nested_detectors

                shield_config["detector_params"] = detector_params

                # Create appropriate detector config
                detector_class = (
                    ChatDetectorConfig if is_chat else ContentDetectorConfig
                )
                self.detectors_internal[shield_id] = detector_class(**shield_config)

    @property
    def all_detectors(
        self,
    ) -> Dict[str, Union[ContentDetectorConfig, ChatDetectorConfig]]:
        """Get all detector configurations"""
        return self.detectors_internal

    # Update other methods to use detectors_internal instead of _detectors
    def get_detectors_by_type(
        self, message_type: Union[str, MessageType]
    ) -> Dict[str, Union[ContentDetectorConfig, ChatDetectorConfig]]:
        """Get detectors for a specific message type"""
        type_value = (
            message_type.value
            if isinstance(message_type, MessageType)
            else message_type
        )
        return {
            shield_id: shield
            for shield_id, shield in self.detectors_internal.items()
            if type_value in shield.message_types
        }

    # Convenience properties
    @property
    def user_message_detectors(self):
        return self.get_detectors_by_type(MessageType.USER)

    @property
    def system_message_detectors(self):
        return self.get_detectors_by_type(MessageType.SYSTEM)

    @property
    def tool_response_detectors(self):
        return self.get_detectors_by_type(MessageType.TOOL)

    @property
    def completion_message_detectors(self):
        return self.get_detectors_by_type(MessageType.COMPLETION)
