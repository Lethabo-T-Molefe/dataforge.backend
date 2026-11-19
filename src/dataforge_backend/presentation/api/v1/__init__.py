"""Version 1 of the DataForge API."""

from fastapi import APIRouter

from . import admin, analysis, auth, cleaning, connectors, datasets, insights, models, platform

router = APIRouter()
# models comes before insights so its /predictions/models/... paths are matched first.
for module in (auth, admin, datasets, cleaning, connectors, analysis, models, insights, platform):
    router.include_router(module.router)
