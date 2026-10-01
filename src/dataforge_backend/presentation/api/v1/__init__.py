"""Version 1 of the DataForge API."""

from fastapi import APIRouter

from . import admin, analysis, auth, cleaning, connectors, datasets, insights, platform

router = APIRouter()
for module in (auth, admin, datasets, cleaning, connectors, analysis, insights, platform):
    router.include_router(module.router)
