from pydantic import BaseModel, ConfigDict, Field, field_validator


class Filters(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ticker: str | None = Field(default=None, max_length=20)
    year: str | None = Field(default=None, max_length=4)
    form: str | None = Field(default=None, max_length=20)
    page: str | None = Field(default=None, max_length=40)

    @field_validator("ticker", "form")
    @classmethod
    def uppercase(cls, value):
        return value.strip().upper() if value else None


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1, max_length=500)
    filters: Filters = Field(default_factory=Filters)
    limit: int = Field(default=5, ge=1, le=20)

    @field_validator("question")
    @classmethod
    def nonempty(cls, value):
        if not value.strip():
            raise ValueError("Question must contain text")
        return value.strip()


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1, max_length=500)
    filters: Filters = Field(default_factory=Filters)

    @field_validator("question")
    @classmethod
    def nonempty(cls, value):
        if not value.strip():
            raise ValueError("Question must contain text")
        return value.strip()
