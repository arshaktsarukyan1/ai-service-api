import math
from datetime import date

import pytest

from app.application.geo_service import (
    build_fences,
    check_proximity,
    distance_meters,
    resolve_radius_meters,
)
from app.domain.geo import GeoCoordinate, GeoEventType, GeoFence, GeoFenceState
from app.domain.location import ConstructionSiteLocation
from app.infrastructure.config_schema import GeoFencingConfig

_EARTH_RADIUS_METERS = 6_371_000


def _coordinate_north_of_equator(
    distance: float,
    *,
    accuracy: float | None = 10,
) -> GeoCoordinate:
    return GeoCoordinate(
        latitude=math.degrees(distance / _EARTH_RADIUS_METERS),
        longitude=0,
        accuracy_meters=accuracy,
    )


def _fence(radius_meters: float = 100) -> GeoFence:
    return GeoFence(
        id="geofence-test-location",
        location_id="test-location",
        name="Test Location",
        center=GeoCoordinate(latitude=0, longitude=0),
        radius_meters=radius_meters,
    )


def _location(
    *,
    location_id: str = "test-location",
    latitude: float | None = 52.525,
    longitude: float | None = 13.3692,
) -> ConstructionSiteLocation:
    return ConstructionSiteLocation(
        id=location_id,
        name="Test Location",
        start_date=date(2026, 1, 1),
        expected_end_date=date(2026, 12, 31),
        description="Test project",
        costs="EUR 1",
        initiator="Test Initiator",
        address="Test Address",
        area="Test Area",
        latitude=latitude,
        longitude=longitude,
    )


def test_distance_meters_returns_zero_for_same_coordinate() -> None:
    coordinate = GeoCoordinate(latitude=52.525, longitude=13.3692)

    assert distance_meters(coordinate, coordinate) == pytest.approx(0)


def test_distance_meters_is_approximately_correct_for_nearby_coordinates() -> None:
    start = GeoCoordinate(latitude=52.525, longitude=13.3692)
    north = GeoCoordinate(latitude=52.526, longitude=13.3692)

    assert distance_meters(start, north) == pytest.approx(111.2, abs=0.2)


def test_distance_meters_is_symmetric() -> None:
    first = GeoCoordinate(latitude=48.1325, longitude=16.321)
    second = GeoCoordinate(latitude=51.9605, longitude=3.6906)

    assert distance_meters(first, second) == pytest.approx(
        distance_meters(second, first),
        abs=0.001,
    )


def test_resolve_radius_uses_default_when_not_provided() -> None:
    assert resolve_radius_meters(None, GeoFencingConfig()) == 100


def test_resolve_radius_accepts_configured_min_and_max() -> None:
    config = GeoFencingConfig(min_radius_meters=10, max_radius_meters=500)

    assert resolve_radius_meters(10, config) == 10
    assert resolve_radius_meters(500, config) == 500


@pytest.mark.parametrize("radius_meters", [0, 9.9, 501])
def test_resolve_radius_rejects_values_outside_configured_bounds(
    radius_meters: float,
) -> None:
    config = GeoFencingConfig(min_radius_meters=10, max_radius_meters=500)

    with pytest.raises(ValueError, match="radius_meters"):
        resolve_radius_meters(radius_meters, config)


def test_build_fences_rejects_zero_radius_when_called_directly() -> None:
    with pytest.raises(ValueError, match="radius_meters"):
        build_fences(
            [_location()],
            radius_meters=0,
            config=GeoFencingConfig(),
        )


def test_build_fences_converts_locations_with_coordinates() -> None:
    fences = build_fences(
        [_location()],
        radius_meters=250,
        config=GeoFencingConfig(),
    )

    assert len(fences) == 1
    assert fences[0].id == "geofence-test-location"
    assert fences[0].location_id == "test-location"
    assert fences[0].center.latitude == 52.525
    assert fences[0].center.longitude == 13.3692
    assert fences[0].radius_meters == 250
    assert fences[0].metadata["address"] == "Test Address"


@pytest.mark.parametrize(
    ("latitude", "longitude"),
    [(None, 13.3692), (52.525, None), (None, None)],
)
def test_build_fences_skips_locations_without_complete_coordinates(
    latitude: float | None,
    longitude: float | None,
) -> None:
    fences = build_fences(
        [_location(latitude=latitude, longitude=longitude)],
        config=GeoFencingConfig(),
    )

    assert fences == []


def test_check_proximity_returns_entered_for_first_inside_detection() -> None:
    events = check_proximity(
        GeoCoordinate(latitude=0, longitude=0, accuracy_meters=5),
        [_fence()],
        config=GeoFencingConfig(),
    )

    assert events[0].event_type is GeoEventType.entered
    assert events[0].trigger_recommended is True
    assert events[0].distance_meters == pytest.approx(0)


def test_check_proximity_treats_exact_radius_boundary_as_entered() -> None:
    events = check_proximity(
        _coordinate_north_of_equator(100),
        [_fence()],
        config=GeoFencingConfig(),
    )

    assert events[0].event_type is GeoEventType.entered
    assert events[0].trigger_recommended is True


def test_check_proximity_returns_inside_without_trigger_for_inside_state() -> None:
    events = check_proximity(
        _coordinate_north_of_equator(50),
        [_fence()],
        previous_states={"geofence-test-location": GeoFenceState.inside},
        config=GeoFencingConfig(),
    )

    assert events[0].event_type is GeoEventType.inside
    assert events[0].trigger_recommended is False


def test_check_proximity_returns_outside_for_first_outside_detection() -> None:
    events = check_proximity(
        _coordinate_north_of_equator(150),
        [_fence()],
        config=GeoFencingConfig(),
    )

    assert events[0].event_type is GeoEventType.outside
    assert events[0].trigger_recommended is False


def test_check_proximity_returns_exited_after_inside_state_beyond_hysteresis() -> None:
    events = check_proximity(
        _coordinate_north_of_equator(126),
        [_fence()],
        previous_states={"geofence-test-location": GeoFenceState.inside},
        config=GeoFencingConfig(exit_hysteresis_meters=25),
    )

    assert events[0].event_type is GeoEventType.exited
    assert events[0].trigger_recommended is False


def test_check_proximity_keeps_inside_state_within_hysteresis_band() -> None:
    events = check_proximity(
        _coordinate_north_of_equator(120),
        [_fence()],
        previous_states={"geofence-test-location": GeoFenceState.inside},
        config=GeoFencingConfig(exit_hysteresis_meters=25),
    )

    assert events[0].event_type is GeoEventType.inside
    assert events[0].trigger_recommended is False


def test_check_proximity_keeps_inside_state_on_exact_hysteresis_boundary() -> None:
    events = check_proximity(
        _coordinate_north_of_equator(125),
        [_fence()],
        previous_states={"geofence-test-location": GeoFenceState.inside},
        config=GeoFencingConfig(exit_hysteresis_meters=25),
    )

    assert events[0].event_type is GeoEventType.inside
    assert events[0].trigger_recommended is False


def test_check_proximity_marks_low_accuracy_location_uncertain() -> None:
    events = check_proximity(
        GeoCoordinate(latitude=0, longitude=0, accuracy_meters=101),
        [_fence()],
        config=GeoFencingConfig(max_acceptable_accuracy_meters=100),
    )

    assert events[0].event_type is GeoEventType.uncertain
    assert events[0].trigger_recommended is False


def test_check_proximity_accepts_exact_maximum_accuracy_for_trigger() -> None:
    events = check_proximity(
        GeoCoordinate(latitude=0, longitude=0, accuracy_meters=100),
        [_fence()],
        config=GeoFencingConfig(max_acceptable_accuracy_meters=100),
    )

    assert events[0].event_type is GeoEventType.entered
    assert events[0].trigger_recommended is True


def test_check_proximity_allows_missing_accuracy_to_classify_position() -> None:
    events = check_proximity(
        GeoCoordinate(latitude=0, longitude=0),
        [_fence()],
        config=GeoFencingConfig(),
    )

    assert events[0].event_type is GeoEventType.entered
    assert events[0].accuracy_meters is None


def test_check_proximity_returns_events_sorted_by_nearest_fence() -> None:
    far_fence = GeoFence(
        id="geofence-far",
        location_id="far",
        name="Far",
        center=_coordinate_north_of_equator(300, accuracy=None),
        radius_meters=100,
    )

    events = check_proximity(
        GeoCoordinate(latitude=0, longitude=0),
        [far_fence, _fence()],
        config=GeoFencingConfig(),
    )

    assert [event.fence.id for event in events] == [
        "geofence-test-location",
        "geofence-far",
    ]
