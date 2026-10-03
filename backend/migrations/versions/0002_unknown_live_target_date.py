"""Allow an unknown future Cotton observation date without inventing a calendar.

Revision ID: 0002
Revises: 0001
"""

import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("forecasts") as batch:
        batch.alter_column("target_date", existing_type=sa.Date(), nullable=True)


def downgrade() -> None:
    # Refuse data loss or fabricated dates when v2 live forecasts exist.
    count = op.get_bind().scalar(sa.text("SELECT COUNT(*) FROM forecasts WHERE target_date IS NULL"))
    if count:
        raise RuntimeError("Cannot restore non-null dates while forecasts with unknown target dates exist")
    with op.batch_alter_table("forecasts") as batch:
        batch.alter_column("target_date", existing_type=sa.Date(), nullable=False)
