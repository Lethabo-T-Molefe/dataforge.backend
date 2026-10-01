"""Operational commands.

dataforge migrate                 apply database migrations
dataforge seed-demo               create the demo organisation through the real pipeline
dataforge create-admin            create or promote an administrator
dataforge evaluate-monitors       evaluate due monitors once (for cron, instead of the in-process scheduler)
dataforge superset-sync           register approved datasets, metrics and row policies in Superset
dataforge serve                   run the API
"""

import asyncio
import getpass
import json
from pathlib import Path
from typing import Any

import typer

from .core.config import settings

app = typer.Typer(help="DataForge backend operations.", no_args_is_help=True)
ROOT = Path(__file__).resolve().parents[2]


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


@app.command()
def migrate(revision: str = "head") -> None:
    """Apply database migrations (creates the app, bronze, silver and gold schemas and the sandbox role)."""
    from alembic import command
    from alembic.config import Config

    config = Config(str(ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(ROOT / "alembic"))
    command.upgrade(config, revision)
    typer.echo(f"Database at {revision}.")


@app.command("seed-demo")
def seed_demo() -> None:
    """Create the demo members, datasets, metrics, dashboards, monitors and a model version."""
    if settings.ENVIRONMENT == "production":
        raise typer.BadParameter("Demo data is not created in production.")

    async def go() -> dict[str, Any]:
        from .core.database import SessionLocal, engine
        from .seed import seed_demo as seed

        try:
            async with SessionLocal() as session:
                return await seed(session)
        finally:
            await engine.dispose()

    result = _run(go())
    typer.echo(json.dumps(result, indent=2))
    typer.echo(
        "Demo members sign in as <name>@dataforge.local with the DEMO_PASSWORD from settings (e.g. thandi@dataforge.local)."
    )


@app.command("create-admin")
def create_admin(
    email: str = typer.Option(..., prompt=True), name: str = typer.Option(..., prompt=True)
) -> None:
    """Create an administrator, or promote an existing member."""
    password = getpass.getpass("Password (leave empty to keep an existing one): ")

    async def go() -> str:
        from sqlalchemy import select

        from .application.identity import check_password_strength, normalise_email
        from .core.database import SessionLocal, engine
        from .core.security import hash_password, utcnow
        from .infrastructure.db.models import User, new_id

        try:
            async with SessionLocal() as session:
                address = normalise_email(email)
                user = (await session.execute(select(User).where(User.email == address))).scalar_one_or_none()
                if user is None:
                    check_password_strength(password, address, name)
                    user = User(
                        id=new_id(),
                        email=address,
                        name=name,
                        role="admin",
                        password_hash=hash_password(password),
                        email_verified_at=utcnow(),
                    )
                    session.add(user)
                    outcome = "created"
                else:
                    user.role = "admin"
                    user.status = "active"
                    if password:
                        check_password_strength(password, address, name)
                        user.password_hash = hash_password(password)
                    outcome = "promoted"
                await session.commit()
                return outcome
        finally:
            await engine.dispose()

    typer.echo(f"Administrator {_run(go())}.")


@app.command("evaluate-monitors")
def evaluate_monitors() -> None:
    """Evaluate every due monitor once and print the outcomes."""

    async def go() -> list[dict[str, Any]]:
        from .application import monitors
        from .core.database import SessionLocal, engine

        try:
            async with SessionLocal() as session:
                return await monitors.evaluate_due(session)
        finally:
            await engine.dispose()

    for result in _run(go()):
        typer.echo(f"{result['monitorId']}: {'ALERT ' if result['triggered'] else ''}{result['message']}")


@app.command("superset-sync")
def superset_sync() -> None:
    """Create the read-only BI login, register the warehouse in Superset, and sync datasets and policies."""

    async def go() -> dict[str, Any]:
        from .application import integrations
        from .core.database import SessionLocal, engine
        from .infrastructure.superset import superset

        try:
            async with SessionLocal() as session:
                return await integrations.sync_all(session)
        finally:
            await superset.close()
            await engine.dispose()

    typer.echo(json.dumps(_run(go()), indent=2))


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000, reload: bool = False) -> None:
    """Run the API with uvicorn."""
    import uvicorn

    uvicorn.run("dataforge_backend.main:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()
