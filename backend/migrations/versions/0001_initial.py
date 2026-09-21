"""Initial CottonLens schema.

Revision ID: 0001
Revises:
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "market_observations",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("observed_on", sa.Date(), nullable=False),
        sa.Column("series", sa.String(32), nullable=False),
        sa.Column("close", sa.Float(), nullable=False),
        sa.Column("source", sa.String(64), nullable=False),
        sa.Column("ingested_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("observed_on", "series", name="uq_observation_date_series"),
    )
    op.create_index("ix_market_observations_observed_on", "market_observations", ["observed_on"])
    op.create_index("ix_market_observations_series", "market_observations", ["series"])
    op.create_table(
        "feature_snapshots",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("as_of_date", sa.Date(), nullable=False, unique=True),
        sa.Column("schema_hash", sa.String(64), nullable=False),
        sa.Column("values", sa.JSON(), nullable=False),
    )
    op.create_index("ix_feature_snapshots_as_of_date", "feature_snapshots", ["as_of_date"])
    op.create_table(
        "model_versions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("artifact_version", sa.String(96), nullable=False),
        sa.Column("horizon", sa.Integer(), nullable=False),
        sa.Column("model_name", sa.String(64), nullable=False),
        sa.Column("model_format", sa.String(32), nullable=False),
        sa.Column("selected", sa.Boolean(), nullable=False),
        sa.Column("data_quality", sa.String(32), nullable=False),
        sa.Column("metrics", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("artifact_version", "horizon", "model_name", name="uq_artifact_horizon_model"),
    )
    op.create_index("ix_model_versions_artifact_version", "model_versions", ["artifact_version"])
    op.create_table(
        "forecasts",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("as_of_date", sa.Date(), nullable=False),
        sa.Column("target_date", sa.Date(), nullable=False),
        sa.Column("horizon", sa.Integer(), nullable=False),
        sa.Column("current_price", sa.Float(), nullable=False),
        sa.Column("predicted_price", sa.Float(), nullable=False),
        sa.Column("predicted_return_pct", sa.Float(), nullable=False),
        sa.Column("actual_price", sa.Float(), nullable=True),
        sa.Column("origin_type", sa.String(16), nullable=False),
        sa.Column("model_version_id", sa.String(36), sa.ForeignKey("model_versions.id"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "as_of_date", "horizon", "origin_type", "model_version_id", name="uq_forecast_identity"
        ),
    )
    op.create_index("ix_forecasts_as_of_date", "forecasts", ["as_of_date"])
    op.create_index("ix_forecasts_horizon", "forecasts", ["horizon"])
    op.create_index("ix_forecasts_origin_type", "forecasts", ["origin_type"])
    op.create_table(
        "forecast_explanations",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("forecast_id", sa.String(36), sa.ForeignKey("forecasts.id"), nullable=False, unique=True),
        sa.Column("base_value", sa.Float(), nullable=False),
        sa.Column("explainer", sa.String(64), nullable=False),
        sa.Column("contributions", sa.JSON(), nullable=False),
    )
    op.create_table(
        "simulations",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("as_of_date", sa.Date(), nullable=False),
        sa.Column("adjustments", sa.JSON(), nullable=False),
        sa.Column("results", sa.JSON(), nullable=False),
        sa.Column("model_version", sa.String(96), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_simulations_as_of_date", "simulations", ["as_of_date"])
    op.create_table(
        "artifact_imports",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("artifact_version", sa.String(96), nullable=False, unique=True),
        sa.Column("manifest", sa.JSON(), nullable=False),
        sa.Column("checksum", sa.String(64), nullable=False),
        sa.Column("imported_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    for table in (
        "artifact_imports",
        "simulations",
        "forecast_explanations",
        "forecasts",
        "model_versions",
        "feature_snapshots",
        "market_observations",
    ):
        op.drop_table(table)
