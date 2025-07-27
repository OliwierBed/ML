# common/db.py
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import text
import logging

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


_engine = None
_Session = None


def init_engine(db_url: str, pool_size: int = 20, max_overflow: int = 20):
    global _engine, _Session
    _engine = create_async_engine(
        db_url,
        echo=False,
        pool_size=pool_size,
        max_overflow=max_overflow,
        future=True,
    )
    _Session = async_sessionmaker(_engine, expire_on_commit=False, class_=AsyncSession)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with _Session() as session:
        yield session


async def check_connection():
    async with _engine.connect() as conn:
        await conn.execute(text("SELECT 1"))
        logger.info("DB connection OK")
