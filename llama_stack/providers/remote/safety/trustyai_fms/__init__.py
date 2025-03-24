import logging
from typing import Any, Dict, Optional, Union

# Remove register_provider import since registration is in registry/safety.py
from llama_stack.apis.safety import Safety
from llama_stack.providers.remote.safety.trustyai_fms.config import (
    ChatDetectorConfig,
    ContentDetectorConfig,
    DetectorParams,
    EndpointType,
    FMSSafetyProviderConfig,
)
from llama_stack.providers.remote.safety.trustyai_fms.detectors.base import (
    BaseDetector,
    DetectorProvider,
)
from llama_stack.providers.remote.safety.trustyai_fms.detectors.chat import ChatDetector
from llama_stack.providers.remote.safety.trustyai_fms.detectors.content import (
    ContentDetector,
)

# Set up logging
logger = logging.getLogger(__name__)

# Type aliases for better readability
ConfigType = Union[ContentDetectorConfig, ChatDetectorConfig, FMSSafetyProviderConfig]
DetectorType = Union[BaseDetector, DetectorProvider]


class DetectorConfigError(ValueError):
    """Raised when detector configuration is invalid"""

    pass


async def create_fms_provider(config: Dict[str, Any]) -> Safety:
    """Create FMS safety provider instance.

    Args:
        config: Configuration dictionary

    Returns:
        Safety: Configured FMS safety provider
    """
    logger.debug("Creating trustyai-fms provider")
    return await get_adapter_impl(FMSSafetyProviderConfig(**config))


async def get_adapter_impl(
    config: Union[Dict[str, Any], FMSSafetyProviderConfig],
    _deps: Optional[Dict[str, Any]] = None,
) -> DetectorType:
    """Get appropriate detector implementation(s) based on config type.

    Args:
        config: Configuration dictionary or FMSSafetyProviderConfig instance
        _deps: Optional dependencies for testing/injection

    Returns:
        Configured detector implementation

    Raises:
        DetectorConfigError: If configuration is invalid
    """
    try:
        if isinstance(config, FMSSafetyProviderConfig):
            provider_config = config
        else:
            provider_config = FMSSafetyProviderConfig(**config)

        detectors: Dict[str, DetectorType] = {}

        # Changed from provider_config.detectors to provider_config.shields
        for shield_id, shield_config in provider_config.shields.items():
            impl: BaseDetector
            if isinstance(shield_config, ChatDetectorConfig):
                impl = ChatDetector(shield_config)
            elif isinstance(shield_config, ContentDetectorConfig):
                impl = ContentDetector(shield_config)
            else:
                raise DetectorConfigError(
                    f"Invalid shield config type for {shield_id}: {type(shield_config)}"
                )
            await impl.initialize()
            detectors[shield_id] = impl

        detectors_for_provider: Dict[str, BaseDetector] = {}
        for shield_id, detector in detectors.items():
            if isinstance(detector, BaseDetector):
                detectors_for_provider[shield_id] = detector

        return DetectorProvider(detectors_for_provider)

    except Exception as e:
        raise DetectorConfigError(
            f"Failed to create detector implementation: {str(e)}"
        ) from e


__all__ = [
    # Factory methods
    "get_adapter_impl",
    "create_fms_provider",
    # Configurations
    "ContentDetectorConfig",
    "ChatDetectorConfig",
    "FMSSafetyProviderConfig",
    "EndpointType",
    "DetectorParams",
    # Implementations
    "ChatDetector",
    "ContentDetector",
    "BaseDetector",
    "DetectorProvider",
    # Types
    "ConfigType",
    "DetectorType",
    "DetectorConfigError",
]
