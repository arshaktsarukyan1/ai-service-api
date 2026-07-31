import math
from datetime import date

from fastapi.testclient import TestClient

from app.domain.location import ConstructionSiteLocation
from app.infrastructure.config_schema import AIProvidersConfig, GeoFencingConfig
from app.infrastructure.yaml_config import get_ai_config
from app.main import create_app

_EARTH_RADIUS_METERS = 6_371_000


def _berlin_coordinate(accuracy_meters: float = 10) -> dict:
    return {
        "latitude": 52.5250,
        "longitude": 13.3692,
        "accuracy_meters": accuracy_meters,
    }


def _coordinate_north_of_berlin(distance_meters: float) -> dict:
    return {
        "latitude": 52.5250 + math.degrees(distance_meters / _EARTH_RADIUS_METERS),
        "longitude": 13.3692,
        "accuracy_meters": 10,
    }


def _provider_config_payload() -> dict:
    return {"api_key_env": "OPENAI_API_KEY", "default_model": "gpt-4o-mini"}


def _location(
    *,
    location_id: str,
    latitude: float | None,
    longitude: float | None,
) -> ConstructionSiteLocation:
    return ConstructionSiteLocation(
        id=location_id,
        name=f"Location {location_id}",
        start_date=date(2026, 1, 1),
        expected_end_date=date(2026, 12, 31),
        description="Test project",
        costs="EUR 1",
        initiator="Test Initiator",
        address="Test Address",
        area=None,
        latitude=latitude,
        longitude=longitude,
    )


def test_geo_config_returns_frontend_safe_defaults(client: TestClient) -> None:
    resp = client.get("/geo/config")

    assert resp.status_code == 200
    body = resp.json()
    assert body == {
        "default_radius_meters": 100,
        "min_radius_meters": 10,
        "max_radius_meters": 5000,
        "exit_hysteresis_meters": 25,
        "trigger_cooldown_seconds": 60,
        "max_acceptable_accuracy_meters": 100,
    }
    assert "X-Request-ID" in resp.headers


def test_geo_fences_returns_known_development_locations(client: TestClient) -> None:
    resp = client.get("/geo/fences")

    assert resp.status_code == 200
    body = resp.json()
    assert body["radius_meters"] == 100
    ids = {fence["location_id"] for fence in body["fences"]}
    assert ids == {
        "at-vienna-bypass-west",
        "de-berlin-hbf-upgrade",
        "nl-rotterdam-port-quay-7",
    }


def test_geo_fences_response_contains_frontend_contract_fields(
    client: TestClient,
) -> None:
    resp = client.get("/geo/fences")

    assert resp.status_code == 200
    fence = resp.json()["fences"][0]
    assert {
        "id",
        "location_id",
        "name",
        "center",
        "radius_meters",
        "metadata",
    }.issubset(fence)
    assert {"latitude", "longitude", "accuracy_meters"}.issubset(fence["center"])
    assert "address" in fence["metadata"]
    assert "area" in fence["metadata"]


def test_geo_fences_accepts_radius_inside_configured_bounds(client: TestClient) -> None:
    resp = client.get("/geo/fences", params={"radius_meters": 250})

    assert resp.status_code == 200
    body = resp.json()
    assert body["radius_meters"] == 250
    assert {fence["radius_meters"] for fence in body["fences"]} == {250}


def test_geo_fences_rejects_radius_below_configured_minimum(client: TestClient) -> None:
    resp = client.get("/geo/fences", params={"radius_meters": 5})

    assert resp.status_code == 422
    body = resp.json()
    assert body["error"] == "geo_fence_validation_error"
    assert "at least 10" in body["detail"]


def test_geo_fences_rejects_radius_above_configured_maximum(client: TestClient) -> None:
    resp = client.get("/geo/fences", params={"radius_meters": 5001})

    assert resp.status_code == 422
    body = resp.json()
    assert body["error"] == "geo_fence_validation_error"
    assert "at most 5000" in body["detail"]


def test_geo_fences_rejects_non_positive_radius_with_request_validation(
    client: TestClient,
) -> None:
    resp = client.get("/geo/fences", params={"radius_meters": 0})

    assert resp.status_code == 422
    assert resp.json()["error"] == "validation_error"


def test_geo_check_returns_entered_event_for_known_location(client: TestClient) -> None:
    resp = client.post(
        "/geo/check",
        json={"user_location": _berlin_coordinate(), "radius_meters": 100},
    )

    assert resp.status_code == 200
    body = resp.json()
    first_event = body["events"][0]
    assert body["radius_meters"] == 100
    assert first_event["fence"]["location_id"] == "de-berlin-hbf-upgrade"
    assert first_event["event_type"] == "entered"
    assert first_event["trigger_recommended"] is True
    assert first_event["distance_meters"] == 0
    assert first_event["accuracy_meters"] == 10


def test_geo_check_returns_no_trigger_for_far_away_coordinate(
    client: TestClient,
) -> None:
    resp = client.post(
        "/geo/check",
        json={
            "user_location": {
                "latitude": 0,
                "longitude": 0,
                "accuracy_meters": 10,
            },
            "radius_meters": 100,
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["events"]
    assert all(event["event_type"] == "outside" for event in body["events"])
    assert all(event["trigger_recommended"] is False for event in body["events"])


def test_geo_check_returns_events_sorted_by_distance(client: TestClient) -> None:
    resp = client.post(
        "/geo/check",
        json={
            "user_location": {
                "latitude": 52.525,
                "longitude": 13.3692,
                "accuracy_meters": 10,
            },
        },
    )

    assert resp.status_code == 200
    events = resp.json()["events"]
    distances = [event["distance_meters"] for event in events]
    assert distances == sorted(distances)


def test_geo_check_response_contains_frontend_contract_fields(
    client: TestClient,
) -> None:
    resp = client.post(
        "/geo/check",
        json={"user_location": _berlin_coordinate()},
    )

    assert resp.status_code == 200
    event = resp.json()["events"][0]
    assert {
        "fence",
        "event_type",
        "distance_meters",
        "radius_meters",
        "accuracy_meters",
        "trigger_recommended",
    }.issubset(event)
    assert event["fence"]["location_id"] == "de-berlin-hbf-upgrade"


def test_geo_check_marks_low_accuracy_location_uncertain(client: TestClient) -> None:
    resp = client.post(
        "/geo/check",
        json={"user_location": _berlin_coordinate(accuracy_meters=101)},
    )

    assert resp.status_code == 200
    first_event = resp.json()["events"][0]
    assert first_event["fence"]["location_id"] == "de-berlin-hbf-upgrade"
    assert first_event["event_type"] == "uncertain"
    assert first_event["trigger_recommended"] is False


def test_geo_check_uses_previous_inside_state_to_emit_inside_not_entered(
    client: TestClient,
) -> None:
    resp = client.post(
        "/geo/check",
        json={
            "user_location": _berlin_coordinate(),
            "previous_states": {"geofence-de-berlin-hbf-upgrade": "inside"},
        },
    )

    assert resp.status_code == 200
    first_event = resp.json()["events"][0]
    assert first_event["event_type"] == "inside"
    assert first_event["trigger_recommended"] is False


def test_geo_check_emits_exited_when_previous_inside_state_moves_beyond_hysteresis(
    client: TestClient,
) -> None:
    resp = client.post(
        "/geo/check",
        json={
            "user_location": _coordinate_north_of_berlin(126),
            "previous_states": {"geofence-de-berlin-hbf-upgrade": "inside"},
        },
    )

    assert resp.status_code == 200
    first_event = resp.json()["events"][0]
    assert first_event["fence"]["location_id"] == "de-berlin-hbf-upgrade"
    assert first_event["event_type"] == "exited"
    assert first_event["trigger_recommended"] is False


def test_geo_check_rejects_radius_above_configured_maximum(client: TestClient) -> None:
    resp = client.post(
        "/geo/check",
        json={"user_location": _berlin_coordinate(), "radius_meters": 5001},
    )

    assert resp.status_code == 422
    body = resp.json()
    assert body["error"] == "geo_fence_validation_error"
    assert "at most 5000" in body["detail"]


def test_geo_check_rejects_invalid_previous_state(client: TestClient) -> None:
    resp = client.post(
        "/geo/check",
        json={
            "user_location": _berlin_coordinate(),
            "previous_states": {"geofence-de-berlin-hbf-upgrade": "nearby"},
        },
    )

    assert resp.status_code == 422
    assert resp.json()["error"] == "validation_error"


def test_geo_check_rejects_invalid_coordinate_payload(client: TestClient) -> None:
    resp = client.post(
        "/geo/check",
        json={
            "user_location": {
                "latitude": 91,
                "longitude": 13.3692,
                "accuracy_meters": 10,
            },
        },
    )

    assert resp.status_code == 422
    assert resp.json()["error"] == "validation_error"


def test_geo_routes_can_override_config_and_skip_locations_without_coordinates(
    monkeypatch,
) -> None:
    from app.interfaces import geo_routes

    config = AIProvidersConfig(
        active_provider="openai",
        providers={"openai": _provider_config_payload()},
        geofencing=GeoFencingConfig(
            default_radius_meters=250,
            min_radius_meters=25,
            max_radius_meters=1000,
            exit_hysteresis_meters=50,
        ),
    )
    monkeypatch.setattr(
        geo_routes,
        "list_construction_sites",
        lambda: [
            _location(location_id="complete", latitude=52.525, longitude=13.3692),
            _location(location_id="missing-latitude", latitude=None, longitude=13.3692),
            _location(location_id="missing-longitude", latitude=52.525, longitude=None),
        ],
    )

    app = create_app()
    app.dependency_overrides[get_ai_config] = lambda: config

    with TestClient(app, raise_server_exceptions=False) as test_client:
        resp = test_client.get("/geo/fences")

    assert resp.status_code == 200
    body = resp.json()
    assert body["radius_meters"] == 250
    assert [fence["location_id"] for fence in body["fences"]] == ["complete"]
