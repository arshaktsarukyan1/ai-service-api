import json
import logging
import re
from typing import Any

from pydantic import ValidationError

from app.application.ai_service import execute_task
from app.domain.exceptions import FaqResponseParseError
from app.domain.faq import FaqItem
from app.domain.location import ConstructionSiteLocation
from app.domain.models import AIResponse
from app.domain.provider import AIProvider
from app.domain.tasks import AITask
from app.infrastructure.config_schema import ProviderConfig

logger = logging.getLogger(__name__)

_FENCE_RE = re.compile(
    r"```(?:json)?\s*([\s\S]*?)```",
    re.IGNORECASE,
)


def _extract_json_blob(raw: str) -> str:
    text = raw.strip()
    m = _FENCE_RE.search(text)
    if m:
        text = m.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return text[start : end + 1]
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end > start:
        return text[start : end + 1]
    raise FaqResponseParseError("Model output did not contain JSON.")


def _coerce_faq_list(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and "faqs" in payload:
        items = payload["faqs"]
    elif isinstance(payload, list):
        items = payload
    else:
        raise FaqResponseParseError(
            "Expected JSON with a 'faqs' array (or a top-level array of objects)."
        )
    if not isinstance(items, list) or not items:
        raise FaqResponseParseError("FAQ list is missing or empty.")
    if not all(isinstance(x, dict) for x in items):
        raise FaqResponseParseError("Each FAQ entry must be an object.")
    return items


def parse_faq_model_content(content: str) -> list[FaqItem]:
    text = content.strip()
    m = _FENCE_RE.search(text)
    if m:
        text = m.group(1).strip()
    data: Any | None = None
    last_json_err: json.JSONDecodeError | None = None
    candidates: list[str] = [text]
    try:
        candidates.append(_extract_json_blob(text))
    except FaqResponseParseError:
        pass
    for candidate in candidates:
        try:
            data = json.loads(candidate)
            break
        except json.JSONDecodeError as exc:
            last_json_err = exc
    if data is None:
        raise FaqResponseParseError(
            f"Invalid JSON in model output: {last_json_err}"
        ) from last_json_err
    raw_items = _coerce_faq_list(data)
    try:
        return [FaqItem.model_validate(x) for x in raw_items]
    except (TypeError, ValidationError) as exc:
        raise FaqResponseParseError(f"FAQ item validation failed: {exc}") from exc


def build_faq_user_prompt(location: ConstructionSiteLocation) -> str:
    intro = (
        "You are generating an FAQ for citizens, commuters, and contractors "
        "about a construction or infrastructure project.\n"
    )
    return f"""{intro}
Project data (JSON, authoritative):
{location.model_dump_json(indent=2)}

Task:
- Propose 5 to 8 distinct, practical question and answer pairs.
- Cover schedule, impact, scope, responsibility, and high-level cost context
  only when grounded in the data.
- If a detail is not in the data, the answer should state that it is not
  specified in the available project record.
- Match the language of the project name and description in the data.

Output rules:
- Respond with a single JSON object only, no markdown fences, no commentary
  before or after the JSON.
- Schema: {{"faqs": [{{"question": "...", "answer": "..."}}]}}
- Each question and answer must be a non-empty string.
"""


async def generate_faqs_for_location(
    location: ConstructionSiteLocation,
    *,
    provider: AIProvider,
    provider_config: ProviderConfig,
) -> tuple[list[FaqItem], AIResponse]:
    input_text = build_faq_user_prompt(location)
    response = await execute_task(
        AITask.faq_generation,
        input_text,
        provider=provider,
        provider_config=provider_config,
        metadata={"location_id": location.id, "task": AITask.faq_generation.value},
    )
    faqs = parse_faq_model_content(response.content)
    logger.info(
        "FAQ generation completed location_id=%s count=%d model=%s",
        location.id,
        len(faqs),
        response.model,
    )
    return faqs, response
