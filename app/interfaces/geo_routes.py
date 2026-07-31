from fastapi import APIRouter, Query

from app.application.geo_service import (
    build_fences,
    check_proximity,
    resolve_radius_meters,
)
from app.domain.exceptions import GeoFenceValidationError
from app.infrastructure.config_schema import AIProvidersConfig
from app.infrastructure.dev_location_repository import list_construction_sites
from app.infrastructure.yaml_config import AiConfigDep
from app.interfaces.schemas import (
    GeoCheckRequest,
    GeoCheckResponse,
    GeoConfigResponse,
    GeoFenceListResponse,
)

router = APIRouter(prefix="/geo", tags=["geo"])


def _resolve_radius_or_raise(
    radius_meters: float | None,
    config: AIProvidersConfig,
) -> float:
    try:
        return resolve_radius_meters(radius_meters, config.geofencing)
    except ValueError as exc:
        raise GeoFenceValidationError(str(exc)) from exc


@router.get(
    "/config",
    response_model=GeoConfigResponse,
    summary="Return frontend-safe geo-fencing configuration",
)
async def get_geo_config(config: AiConfigDep) -> GeoConfigResponse:
    return GeoConfigResponse.model_validate(config.geofencing.model_dump())


@router.get(
    "/fences",
    response_model=GeoFenceListResponse,
    summary="List configured geo-fences derived from development location data",
)
async def list_geo_fences(
    config: AiConfigDep,
    radius_meters: float | None = Query(default=None, gt=0),
) -> GeoFenceListResponse:
    radius = _resolve_radius_or_raise(radius_meters, config)
    fences = build_fences(
        list_construction_sites(),
        radius_meters=radius,
        config=config.geofencing,
    )
    return GeoFenceListResponse(radius_meters=radius, fences=fences)


@router.post(
    "/check",
    response_model=GeoCheckResponse,
    summary="Check a developer-provided coordinate against configured geo-fences",
)
async def check_geo_proximity(
    request: GeoCheckRequest,
    config: AiConfigDep,
) -> GeoCheckResponse:
    radius = _resolve_radius_or_raise(request.radius_meters, config)
    fences = build_fences(
        list_construction_sites(),
        radius_meters=radius,
        config=config.geofencing,
    )
    events = check_proximity(
        request.user_location,
        fences,
        previous_states=request.previous_states,
        config=config.geofencing,
    )
    return GeoCheckResponse(radius_meters=radius, events=events)
