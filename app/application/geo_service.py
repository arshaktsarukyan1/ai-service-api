import math
from collections.abc import Iterable, Mapping

from app.domain.geo import (
    GeoCoordinate,
    GeoEvent,
    GeoEventType,
    GeoFence,
    GeoFenceState,
)
from app.domain.location import ConstructionSiteLocation
from app.infrastructure.config_schema import GeoFencingConfig

_EARTH_RADIUS_METERS = 6_371_000


def distance_meters(a: GeoCoordinate, b: GeoCoordinate) -> float:
    lat_a = math.radians(a.latitude)
    lat_b = math.radians(b.latitude)
    delta_lat = math.radians(b.latitude - a.latitude)
    delta_lon = math.radians(b.longitude - a.longitude)

    haversine = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat_a) * math.cos(lat_b) * math.sin(delta_lon / 2) ** 2
    )
    central_angle = 2 * math.atan2(math.sqrt(haversine), math.sqrt(1 - haversine))
    return _EARTH_RADIUS_METERS * central_angle


def resolve_radius_meters(
    radius_meters: float | None,
    config: GeoFencingConfig,
) -> float:
    radius = (
        float(config.default_radius_meters)
        if radius_meters is None
        else float(radius_meters)
    )
    if radius < config.min_radius_meters:
        raise ValueError(
            f"radius_meters must be at least {config.min_radius_meters}."
        )
    if radius > config.max_radius_meters:
        raise ValueError(
            f"radius_meters must be at most {config.max_radius_meters}."
        )
    return radius


def build_fences(
    locations: Iterable[ConstructionSiteLocation],
    *,
    radius_meters: float | None = None,
    config: GeoFencingConfig,
) -> list[GeoFence]:
    radius = resolve_radius_meters(radius_meters, config)
    fences: list[GeoFence] = []
    for location in locations:
        if location.latitude is None or location.longitude is None:
            continue
        fences.append(
            GeoFence(
                id=f"geofence-{location.id}",
                location_id=location.id,
                name=location.name,
                center=GeoCoordinate(
                    latitude=location.latitude,
                    longitude=location.longitude,
                ),
                radius_meters=radius,
                metadata={
                    "address": location.address,
                    "area": location.area,
                },
            )
        )
    return fences


def _classify_event(
    *,
    distance: float,
    fence: GeoFence,
    previous_state: GeoFenceState,
    user_location: GeoCoordinate,
    config: GeoFencingConfig,
) -> GeoEvent:
    accuracy = user_location.accuracy_meters
    if accuracy is not None and accuracy > config.max_acceptable_accuracy_meters:
        return GeoEvent(
            fence=fence,
            event_type=GeoEventType.uncertain,
            distance_meters=distance,
            radius_meters=fence.radius_meters,
            accuracy_meters=accuracy,
            trigger_recommended=False,
        )

    if previous_state is GeoFenceState.inside:
        if distance <= fence.radius_meters + config.exit_hysteresis_meters:
            return GeoEvent(
                fence=fence,
                event_type=GeoEventType.inside,
                distance_meters=distance,
                radius_meters=fence.radius_meters,
                accuracy_meters=accuracy,
                trigger_recommended=False,
            )
        return GeoEvent(
            fence=fence,
            event_type=GeoEventType.exited,
            distance_meters=distance,
            radius_meters=fence.radius_meters,
            accuracy_meters=accuracy,
            trigger_recommended=False,
        )

    if distance <= fence.radius_meters:
        return GeoEvent(
            fence=fence,
            event_type=GeoEventType.entered,
            distance_meters=distance,
            radius_meters=fence.radius_meters,
            accuracy_meters=accuracy,
            trigger_recommended=True,
        )

    return GeoEvent(
        fence=fence,
        event_type=GeoEventType.outside,
        distance_meters=distance,
        radius_meters=fence.radius_meters,
        accuracy_meters=accuracy,
        trigger_recommended=False,
    )


def check_proximity(
    user_location: GeoCoordinate,
    fences: Iterable[GeoFence],
    *,
    previous_states: Mapping[str, GeoFenceState] | None = None,
    config: GeoFencingConfig,
) -> list[GeoEvent]:
    states = previous_states or {}
    events = [
        _classify_event(
            distance=distance_meters(user_location, fence.center),
            fence=fence,
            previous_state=states.get(fence.id, GeoFenceState.outside),
            user_location=user_location,
            config=config,
        )
        for fence in fences
    ]
    return sorted(events, key=lambda event: event.distance_meters)
