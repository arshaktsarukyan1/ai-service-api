from datetime import date

from pydantic import BaseModel, ConfigDict, Field


class ConstructionSiteLocation(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = Field(description="Stable identifier (used in API paths).")
    name: str
    start_date: date
    expected_end_date: date
    description: str
    costs: str = Field(
        description=(
            "Reported or estimated project cost (free-text, e.g. currency + amount)."
        ),
    )
    initiator: str
    address: str
    area: str | None = Field(
        default=None, description="Region, district, or site area description."
    )
    latitude: float | None = Field(default=None, description="WGS-84 latitude.")
    longitude: float | None = Field(default=None, description="WGS-84 longitude.")
