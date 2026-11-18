from adapters_v2.groq_adapter import groq_adapter
from adapters_v2.gemini_adapter import gemini_adapter


PROVIDER_ADAPTERS = {
    "groq": groq_adapter,
    "google": gemini_adapter,
}