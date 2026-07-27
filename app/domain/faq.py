from pydantic import BaseModel, ConfigDict, Field


class FaqItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
