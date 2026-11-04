"""Store market prices under their verified artifact import.

Revision ID: 0003
Revises: 0002
"""

import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "artifact_market_observations",
        sa.Column("artifact_import_id", sa.String(length=36), nullable=False),
        sa.Column("observed_on", sa.Date(), nullable=False),
        sa.Column("series", sa.String(length=32), nullable=False),
        sa.Column("close", sa.Float(), nullable=False),
        sa.ForeignKeyConstraint(["artifact_import_id"], ["artifact_imports.id"]),
        sa.PrimaryKeyConstraint("artifact_import_id", "observed_on", "series"),
    )


def downgrade() -> None:
    count = op.get_bind().scalar(sa.text("SELECT COUNT(*) FROM artifact_market_observations"))
    if count:
        raise RuntimeError("Cannot downgrade while artifact-scoped market observations exist")
    op.drop_table("artifact_market_observations")
