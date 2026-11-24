"""Route registration for the /api/v1 prefix."""

from fastapi import APIRouter

from app.api.routes.documents import router as documents_router
from app.api.routes.evaluations import router as evaluations_router
from app.api.routes.events import router as events_router
from app.api.routes.health import router as health_router
from app.api.routes.llm import router as llm_router
from app.api.routes.projects import router as projects_router
from app.api.routes.queries import router as queries_router
from app.api.routes.reports import router as reports_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(llm_router)
api_router.include_router(projects_router)
api_router.include_router(documents_router)
api_router.include_router(queries_router)
api_router.include_router(events_router)
api_router.include_router(evaluations_router)
api_router.include_router(reports_router)
