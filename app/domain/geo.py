from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class GeoFenceState(StrEnum):
    outside = "outside"
    inside = "inside"


class GeoEventType(StrEnum):
    entered = "entered"
    exited = "exited"
    inside = "inside"
    outside = "outside"
    uncertain = "uncertain"


class GeoCoordinate(BaseModel):
    model_config = ConfigDict(frozen=True)

    latitude: float = Field(ge=-90.0, le=90.0)
    longitude: float = Field(ge=-180.0, le=180.0)
    accuracy_meters: float | None = Field(default=None, ge=0.0)


class GeoFence(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=1)
    location_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    center: GeoCoordinate
    radius_meters: float = Field(gt=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class GeoEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    fence: GeoFence
    event_type: GeoEventType
    distance_meters: float = Field(ge=0.0)
    radius_meters: float = Field(gt=0.0)
    accuracy_meters: float | None = Field(default=None, ge=0.0)
    trigger_recommended: bool = False
