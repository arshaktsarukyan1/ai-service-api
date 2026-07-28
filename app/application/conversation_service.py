import logging

from app.application.ai_service import execute_task
from app.domain.location import ConstructionSiteLocation
from app.domain.models import AIResponse
from app.domain.provider import AIProvider
from app.domain.tasks import AITask
from app.domain.voice import Transcript, VoiceSession
from app.infrastructure.config_schema import ProviderConfig

logger = logging.getLogger(__name__)


def build_voice_assistant_prompt(
    transcript: Transcript,
    session: VoiceSession,
    *,
    location: ConstructionSiteLocation | None = None,
) -> str:
    trigger = session.trigger
    location_block = (
        f"""
Location context (authoritative JSON):
{location.model_dump_json(indent=2)}
"""
        if location
        else "Location context: no location record was provided.\n"
    )
    return f"""You are a concise German-first voice assistant for a construction and
infrastructure app.

User transcript:
{transcript.text}

Voice session:
- session_id: {session.id}
- requested_language: {session.language}
- trigger_source: {trigger.source.value}
- event_type: {trigger.event_type or "none"}
- location_id: {trigger.location_id or "none"}

{location_block}
Instructions:
- Reply in German unless the user clearly asks for another language.
- Keep the response suitable for spoken playback: short, direct, and natural.
- If a requested project detail is missing from the provided context, say it is
  not available.
- For app-triggered or proximity-triggered events, proactively explain why the
  user is hearing the alert.
- Do not include markdown, JSON, bullets, or technical metadata in the spoken answer.
"""


async def generate_voice_response(
    transcript: Transcript,
    session: VoiceSession,
    *,
    provider: AIProvider,
    provider_config: ProviderConfig,
    location: ConstructionSiteLocation | None = None,
) -> AIResponse:
    prompt = build_voice_assistant_prompt(
        transcript,
        session,
        location=location,
    )
    response = await execute_task(
        AITask.voice_assistant,
        prompt,
        provider=provider,
        provider_config=provider_config,
        metadata={
            "language": session.language,
            "location_id": session.trigger.location_id,
            "session_id": session.id,
            "trigger_source": session.trigger.source.value,
            "event_type": session.trigger.event_type,
            "task": AITask.voice_assistant.value,
        },
    )
    logger.info(
        "Voice response generated session_id=%s trigger=%s model=%s",
        session.id,
        session.trigger.source.value,
        response.model,
    )
    return response
