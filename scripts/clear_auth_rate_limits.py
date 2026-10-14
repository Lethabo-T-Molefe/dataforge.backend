import asyncio

from sqlalchemy import text

from dataforge_backend.core.database import SessionLocal


async def main() -> None:
    async with SessionLocal() as session:
        await session.execute(text("DELETE FROM app.rate_buckets WHERE key LIKE 'auth:%'"))
        await session.commit()
        print("cleared")


if __name__ == "__main__":
    asyncio.run(main())
