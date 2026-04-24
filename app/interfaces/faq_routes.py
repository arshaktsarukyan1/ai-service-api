from fastapi import APIRouter

from app.application.faq_service import generate_faqs_for_location
from app.domain.exceptions import LocationNotFoundError
from app.infrastructure.dev_location_repository import (
    get_construction_site,
    list_construction_site_ids,
    list_construction_sites,
)
from app.infrastructure.openai_provider import get_active_provider
from app.infrastructure.yaml_config import AiConfigDep
from app.interfaces.schemas import (
    DevLocationListItem,
    DevLocationListResponse,
    FaqByLocationResponse,
    FaqLocationSummary,
    FaqMetadata,
)

router = APIRouter(prefix="/faq", tags=["faq"])


@router.get(
    "/locations",
    response_model=DevLocationListResponse,
    summary="List available development location identifiers",
)
async def list_locations() -> DevLocationListResponse:
    sites = list_construction_sites()
    return DevLocationListResponse(
        locations=[DevLocationListItem(id=s.id, name=s.name) for s in sites]
    )


@router.get(
    "/{location_id}",
    response_model=FaqByLocationResponse,
    summary="Generate FAQs for a location",
)
async def get_faqs_for_location(
    location_id: str,
    config: AiConfigDep,
) -> FaqByLocationResponse:
    location = get_construction_site(location_id)
    if location is None:
        known = ", ".join(list_construction_site_ids()) or "(none)"
        raise LocationNotFoundError(
            f"Unknown location_id={location_id!r}. Known dev ids: {known}"
        )

    provider = get_active_provider(config)
    provider_config = config.providers[config.active_provider]
    faqs, ai = await generate_faqs_for_location(
        location,
        provider=provider,
        provider_config=provider_config,
    )
    return FaqByLocationResponse(
        location=FaqLocationSummary.from_domain(location),
        faqs=faqs,
        generation=FaqMetadata(
            model=ai.model,
            provider=ai.provider,
            latency_ms=ai.latency_ms,
            source="ai_generated",
            total_tokens=ai.usage.total_tokens if ai.usage else None,
        ),
    )
