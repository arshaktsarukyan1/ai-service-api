import pytest
from pydantic import ValidationError

from app.domain.geo import (
    GeoCoordinate,
    GeoEvent,
    GeoEventType,
    GeoFence,
    GeoFenceState,
)


def _valid_coordinate() -> GeoCoordinate:
    return GeoCoordinate(
        latitude=52.525,
        longitude=13.3692,
        accuracy_meters=12.5,
    )


def _valid_fence() -> GeoFence:
    return GeoFence(
        id="fence-de-berlin-hbf-upgrade",
        location_id="de-berlin-hbf-upgrade",
        name="Berlin Central Station",
        center=_valid_coordinate(),
        radius_meters=100,
        metadata={"source": "dev_data"},
    )


def test_geo_coordinate_accepts_valid_values() -> None:
    coordinate = _valid_coordinate()

    assert coordinate.latitude == 52.525
    assert coordinate.longitude == 13.3692
    assert coordinate.accuracy_meters == 12.5


@pytest.mark.parametrize("latitude", [-90.1, 90.1])
def test_geo_coordinate_rejects_latitude_outside_world_bounds(latitude: float) -> None:
    with pytest.raises(ValidationError):
        GeoCoordinate(latitude=latitude, longitude=13.3692)


@pytest.mark.parametrize("longitude", [-180.1, 180.1])
def test_geo_coordinate_rejects_longitude_outside_world_bounds(
    longitude: float,
) -> None:
    with pytest.raises(ValidationError):
        GeoCoordinate(latitude=52.525, longitude=longitude)


@pytest.mark.parametrize("accuracy_meters", [-0.1, -1])
def test_geo_coordinate_rejects_negative_accuracy(accuracy_meters: float) -> None:
    with pytest.raises(ValidationError):
        GeoCoordinate(
            latitude=52.525,
            longitude=13.3692,
            accuracy_meters=accuracy_meters,
        )


@pytest.mark.parametrize(
    ("latitude", "longitude"),
    [(-90.0, -180.0), (90.0, 180.0), (0.0, 0.0)],
)
def test_geo_coordinate_accepts_boundary_values(
    latitude: float,
    longitude: float,
) -> None:
    coordinate = GeoCoordinate(latitude=latitude, longitude=longitude)

    assert coordinate.latitude == latitude
    assert coordinate.longitude == longitude


def test_geo_coordinate_allows_missing_accuracy() -> None:
    coordinate = GeoCoordinate(latitude=52.525, longitude=13.3692)

    assert coordinate.accuracy_meters is None


def test_geo_fence_accepts_valid_values() -> None:
    fence = _valid_fence()

    assert fence.id == "fence-de-berlin-hbf-upgrade"
    assert fence.location_id == "de-berlin-hbf-upgrade"
    assert fence.radius_meters == 100
    assert fence.metadata == {"source": "dev_data"}


@pytest.mark.parametrize("field_name", ["id", "location_id", "name"])
def test_geo_fence_rejects_empty_required_text(field_name: str) -> None:
    data = _valid_fence().model_dump()
    data[field_name] = ""

    with pytest.raises(ValidationError):
        GeoFence.model_validate(data)


@pytest.mark.parametrize("radius_meters", [0, -1, -0.1])
def test_geo_fence_rejects_non_positive_radius(radius_meters: float) -> None:
    with pytest.raises(ValidationError):
        GeoFence(
            id="fence",
            location_id="location",
            name="Location",
            center=_valid_coordinate(),
            radius_meters=radius_meters,
        )


def test_geo_fence_metadata_defaults_to_independent_dicts() -> None:
    first = GeoFence(
        id="first",
        location_id="location-1",
        name="First",
        center=_valid_coordinate(),
        radius_meters=100,
    )
    second = GeoFence(
        id="second",
        location_id="location-2",
        name="Second",
        center=_valid_coordinate(),
        radius_meters=200,
    )

    assert first.metadata == {}
    assert second.metadata == {}
    assert first.metadata is not second.metadata


def test_geo_event_accepts_valid_values() -> None:
    event = GeoEvent(
        fence=_valid_fence(),
        event_type=GeoEventType.entered,
        distance_meters=42.5,
        radius_meters=100,
        accuracy_meters=8,
        trigger_recommended=True,
    )

    assert event.event_type is GeoEventType.entered
    assert event.distance_meters == 42.5
    assert event.trigger_recommended is True


@pytest.mark.parametrize("distance_meters", [-0.1, -1])
def test_geo_event_rejects_negative_distance(distance_meters: float) -> None:
    with pytest.raises(ValidationError):
        GeoEvent(
            fence=_valid_fence(),
            event_type=GeoEventType.inside,
            distance_meters=distance_meters,
            radius_meters=100,
        )


@pytest.mark.parametrize("radius_meters", [0, -1])
def test_geo_event_rejects_non_positive_radius(radius_meters: float) -> None:
    with pytest.raises(ValidationError):
        GeoEvent(
            fence=_valid_fence(),
            event_type=GeoEventType.inside,
            distance_meters=10,
            radius_meters=radius_meters,
        )


def test_geo_event_rejects_negative_accuracy() -> None:
    with pytest.raises(ValidationError):
        GeoEvent(
            fence=_valid_fence(),
            event_type=GeoEventType.inside,
            distance_meters=10,
            radius_meters=100,
            accuracy_meters=-1,
        )


def test_geo_enums_use_stable_wire_values() -> None:
    assert GeoFenceState.outside.value == "outside"
    assert GeoFenceState.inside.value == "inside"
    assert GeoEventType.entered.value == "entered"
    assert GeoEventType.exited.value == "exited"
    assert GeoEventType.inside.value == "inside"
    assert GeoEventType.outside.value == "outside"
    assert GeoEventType.uncertain.value == "uncertain"
