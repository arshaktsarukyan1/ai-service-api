from pathlib import Path

_README = Path(__file__).resolve().parents[2] / "README.md"


def _readme_text() -> str:
    return _README.read_text(encoding="utf-8")


def test_readme_documents_geo_fencing_endpoints() -> None:
    text = _readme_text()

    assert "GET /geo/config" in text
    assert "GET /geo/fences" in text
    assert "POST /geo/check" in text
    assert "curl http://127.0.0.1:8000/geo/config" in text
    assert 'curl "http://127.0.0.1:8000/geo/fences?radius_meters=250"' in text


def test_readme_documents_geo_fencing_configuration() -> None:
    text = _readme_text()

    assert "Geo-fencing settings are in the `geofencing` block" in text
    assert "`default_radius_meters`" in text
    assert "`min_radius_meters`" in text
    assert "`max_radius_meters`" in text
    assert "`exit_hysteresis_meters`" in text
    assert "`trigger_cooldown_seconds`" in text
    assert "`max_acceptable_accuracy_meters`" in text


def test_readme_documents_geo_event_values_and_test_commands() -> None:
    text = _readme_text()

    for event_type in ["entered", "inside", "outside", "exited", "uncertain"]:
        assert f"`{event_type}`" in text
    assert "tests/unit/test_geo_domain.py" in text
    assert "tests/unit/test_geo_service.py" in text
    assert "tests/integration/test_geo_routes.py" in text


def test_readme_keeps_geo_fencing_v1_non_persistent() -> None:
    text = _readme_text()

    assert "No database persistence or migrations are required" in text
    assert "There are no migrations to run yet." in text
