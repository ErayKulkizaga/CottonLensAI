import uuid
from datetime import UTC, date, datetime

from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


def utcnow() -> datetime:
    return datetime.now(UTC)


class MarketObservation(Base):
    __tablename__ = "market_observations"
    __table_args__ = (UniqueConstraint("observed_on", "series", name="uq_observation_date_series"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    observed_on: Mapped[date] = mapped_column(Date, index=True)
    series: Mapped[str] = mapped_column(String(32), index=True)
    close: Mapped[float] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(64))
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class FeatureSnapshot(Base):
    __tablename__ = "feature_snapshots"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    as_of_date: Mapped[date] = mapped_column(Date, unique=True, index=True)
    schema_hash: Mapped[str] = mapped_column(String(64))
    values: Mapped[dict] = mapped_column(JSON)


class ModelVersion(Base):
    __tablename__ = "model_versions"
    __table_args__ = (
        UniqueConstraint("artifact_version", "horizon", "model_name", name="uq_artifact_horizon_model"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    artifact_version: Mapped[str] = mapped_column(String(96), index=True)
    horizon: Mapped[int] = mapped_column(Integer)
    model_name: Mapped[str] = mapped_column(String(64))
    model_format: Mapped[str] = mapped_column(String(32))
    selected: Mapped[bool] = mapped_column(Boolean, default=False)
    data_quality: Mapped[str] = mapped_column(String(32), default="validated")
    metrics: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Forecast(Base):
    __tablename__ = "forecasts"
    __table_args__ = (
        UniqueConstraint(
            "as_of_date", "horizon", "origin_type", "model_version_id", name="uq_forecast_identity"
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    as_of_date: Mapped[date] = mapped_column(Date, index=True)
    target_date: Mapped[date] = mapped_column(Date)
    horizon: Mapped[int] = mapped_column(Integer, index=True)
    current_price: Mapped[float] = mapped_column(Float)
    predicted_price: Mapped[float] = mapped_column(Float)
    predicted_return_pct: Mapped[float] = mapped_column(Float)
    actual_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    origin_type: Mapped[str] = mapped_column(String(16), index=True)
    model_version_id: Mapped[str] = mapped_column(ForeignKey("model_versions.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    model_version: Mapped[ModelVersion] = relationship()
    explanation: Mapped["ForecastExplanation | None"] = relationship(
        back_populates="forecast", uselist=False, cascade="all, delete-orphan"
    )


class ForecastExplanation(Base):
    __tablename__ = "forecast_explanations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    forecast_id: Mapped[str] = mapped_column(ForeignKey("forecasts.id"), unique=True)
    base_value: Mapped[float] = mapped_column(Float)
    explainer: Mapped[str] = mapped_column(String(64))
    contributions: Mapped[list] = mapped_column(JSON)

    forecast: Mapped[Forecast] = relationship(back_populates="explanation")


class Simulation(Base):
    __tablename__ = "simulations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    as_of_date: Mapped[date] = mapped_column(Date, index=True)
    adjustments: Mapped[dict] = mapped_column(JSON)
    results: Mapped[list] = mapped_column(JSON)
    model_version: Mapped[str] = mapped_column(String(96))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ArtifactImport(Base):
    __tablename__ = "artifact_imports"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    artifact_version: Mapped[str] = mapped_column(String(96), unique=True)
    manifest: Mapped[dict] = mapped_column(JSON)
    checksum: Mapped[str] = mapped_column(String(64))
    imported_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
