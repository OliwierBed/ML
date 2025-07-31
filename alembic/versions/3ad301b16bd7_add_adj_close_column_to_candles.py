"""add adj_close column to candles

Revision ID: 3ad301b16bd7
Revises: c351c8a7490b
Create Date: 2025-07-31 01:37:41.954656

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '3ad301b16bd7'
down_revision: Union[str, None] = 'c351c8a7490b'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    op.add_column('candles', sa.Column('adj_close', sa.Float(), nullable=True))

def downgrade():
    op.drop_column('candles', 'adj_close')