from __future__ import annotations

import datetime as _dt

from pydantic import BaseModel, Field, field_validator


class RiskRequest(BaseModel):
    lat: float = Field(ge=-90.0, le=90.0, description="Query latitude")
    lon: float = Field(ge=-180.0, le=180.0, description="Query longitude")
    datetime: _dt.datetime = Field(description="ISO 8601 local timestamp")
    model: str = Field(default="default", description="model identifier")

    @field_validator("lat")
    @classmethod
    def _lat(cls, v):
        if not -90.0 <= v <= 90.0:
            raise ValueError("lat out of range [-90, 90]")
        return v

    @field_validator("lon")
    @classmethod
    def _lon(cls, v):
        if not -180.0 <= v <= 180.0:
            raise ValueError("lon out of range [-180, 180]")
        return v


class RiskResponse(BaseModel):
    risk_score: float
    level: str
    model_used: str
    model_version: str
    last_updated: str
    disclaimer: str


class CompareEntry(BaseModel):
    model_used: str
    model_version: str
    risk_score: float
    level: str


class CompareResponse(BaseModel):
    lat: float
    lon: float
    datetime: str
    estimates: list[CompareEntry]


class VersionInfo(BaseModel):
    version: str
    saved_at: str
    metrics: dict


class ModelInfo(BaseModel):
    name: str
    active_version: str
    versions: list[str]
    last_metrics: dict


class HealthResponse(BaseModel):
    status: str
    models_loaded: int
    active: dict


DISCLAIMER = ("Estimate based on historical patterns, not a prediction of certainty.")


class RouteRequest(BaseModel):
    lat1: float = Field(ge=-90.0, le=90.0, description="Origin latitude")
    lon1: float = Field(ge=-180.0, le=180.0, description="Origin longitude")
    lat2: float = Field(ge=-90.0, le=90.0, description="Destination latitude")
    lon2: float = Field(ge=-180.0, le=180.0, description="Destination longitude")
    datetime: _dt.datetime = Field(description="ISO 8601 local timestamp")


class RoutePoint(BaseModel):
    lat: float
    lon: float


class RouteResponse(BaseModel):
    route: list[RoutePoint]
    risk_score_mean: float
    risk_score_max: float
    disclaimer: str