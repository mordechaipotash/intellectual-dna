"""Dashboard HTML page routes."""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["pages"])


@router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Dashboard home page."""
    templates = request.app.state.templates

    # Check if onboarding is needed
    from brain_mcp.config import get_config
    try:
        cfg = get_config()
        if not cfg.parquet_path.exists():
            return templates.TemplateResponse("onboarding.html", {
                "request": request,
                "active_page": "home",
            })
    except Exception:
        pass

    return templates.TemplateResponse("home.html", {
        "request": request,
        "active_page": "home",
    })


@router.get("/search", response_class=HTMLResponse)
async def search_page(request: Request):
    """Search page."""
    templates = request.app.state.templates
    return templates.TemplateResponse("search.html", {
        "request": request,
        "active_page": "search",
    })


@router.get("/sources", response_class=HTMLResponse)
async def sources_page(request: Request):
    """Sources management page."""
    templates = request.app.state.templates
    return templates.TemplateResponse("sources.html", {
        "request": request,
        "active_page": "sources",
    })


@router.get("/tools", response_class=HTMLResponse)
async def tools_page(request: Request):
    """Tool status page."""
    templates = request.app.state.templates
    return templates.TemplateResponse("tools.html", {
        "request": request,
        "active_page": "tools",
    })


@router.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    """Settings page."""
    templates = request.app.state.templates
    return templates.TemplateResponse("settings.html", {
        "request": request,
        "active_page": "settings",
    })
