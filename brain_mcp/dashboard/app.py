"""
brain-mcp dashboard — FastAPI application factory.

Creates the dashboard app with all routes, static files, and templates.
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

DASHBOARD_DIR = Path(__file__).parent
STATIC_DIR = DASHBOARD_DIR / "static"
TEMPLATE_DIR = DASHBOARD_DIR / "templates"


def create_app() -> FastAPI:
    """Create and configure the Brain MCP Dashboard FastAPI app."""
    app = FastAPI(
        title="Brain MCP Dashboard",
        description="Local web UI for managing your brain",
        docs_url=None,  # No Swagger UI needed
        redoc_url=None,
    )

    # Static files (htmx, Pico CSS, Alpine.js, custom CSS/JS)
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # Templates
    templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

    # Register route modules
    from .routes import pages, stats, search, sources, tools, settings, onboarding, tasks

    app.include_router(pages.router)
    app.include_router(stats.router, prefix="/api/stats")
    app.include_router(search.router, prefix="/api")
    app.include_router(sources.router, prefix="/api/sources")
    app.include_router(tools.router, prefix="/api/tools")
    app.include_router(settings.router, prefix="/api/settings")
    app.include_router(onboarding.router, prefix="/api/onboarding")
    app.include_router(tasks.router, prefix="/api/tasks")

    # Share templates with route modules
    app.state.templates = templates

    return app
