class AIServiceError(Exception):
    """Base exception for all AI service errors."""


class AIProviderError(AIServiceError):
    """Provider-level error (connection failure, bad request, upstream error)."""


class AIAuthError(AIProviderError):
    """API key missing, invalid, or lacking permission."""


class AIRateLimitError(AIProviderError):
    """Provider rate limit or quota exceeded."""


class AITimeoutError(AIProviderError):
    """Provider call exceeded the configured timeout."""


class AIUnsupportedTaskError(AIServiceError):
    """Requested AITask is not supported by the active provider."""


class LocationNotFoundError(AIServiceError):
    """No construction-site / project record exists for the given identifier."""


class GeoFenceValidationError(AIServiceError):
    """Geo-fencing request values are invalid or outside configured limits."""


class FaqResponseParseError(AIServiceError):
    """The AI model output could not be parsed into structured FAQ items."""


class VoiceServiceError(AIServiceError):
    """Base exception for voice interaction errors."""


class VoiceAudioValidationError(VoiceServiceError):
    """Audio input is missing, too large, malformed, or uses an unsupported format."""


class VoiceTranscriptionError(VoiceServiceError):
    """Speech-to-text processing failed."""


class VoiceSynthesisError(VoiceServiceError):
    """Text-to-speech processing failed."""


class VoiceSessionError(VoiceServiceError):
    """Voice session state or protocol handling failed."""
