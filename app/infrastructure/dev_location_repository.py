import json
from functools import lru_cache
from pathlib import Path

from app.domain.location import ConstructionSiteLocation

_DATA_FILE = Path(__file__).resolve().parent / "data" / "dev_construction_sites.json"


@lru_cache(maxsize=1)
def _load_sites() -> dict[str, ConstructionSiteLocation]:
    raw = json.loads(_DATA_FILE.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("dev_construction_sites.json must contain a JSON array")
    by_id: dict[str, ConstructionSiteLocation] = {}
    for item in raw:
        site = ConstructionSiteLocation.model_validate(item)
        if site.id in by_id:
            raise ValueError(f"Duplicate construction site id: {site.id!r}")
        by_id[site.id] = site
    return by_id


def get_construction_site(location_id: str) -> ConstructionSiteLocation | None:
    return _load_sites().get(location_id)


def list_construction_site_ids() -> list[str]:
    return sorted(_load_sites().keys())


def list_construction_sites() -> list[ConstructionSiteLocation]:
    by_id = _load_sites()
    return [by_id[k] for k in sorted(by_id)]
