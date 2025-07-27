import asyncio
from hydra import compose, initialize
from common.db import init_engine, check_connection
import os
from dotenv import load_dotenv

async def main():
    load_dotenv()
    with initialize(version_base=None, config_path="../config"):
        cfg = compose(config_name="config")
    init_engine(cfg.db.url, cfg.db.pool_size, cfg.db.max_overflow)
    await check_connection()

if __name__ == "__main__":
    asyncio.run(main())
