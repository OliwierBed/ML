"""add inserted_at column

Revision ID: f061a3794865
Revises: 3ad301b16bd7
Create Date: 2025-07-31 02:02:14.854345

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f061a3794865'
down_revision: Union[str, None] = '3ad301b16bd7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    op.add_column('candles', sa.Column('inserted_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('NOW()')))

def downgrade():
    op.drop_column('candles', 'inserted_at')
