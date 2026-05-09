"""Dashboard routes — serves the HTML dashboard."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

router = APIRouter()

DASHBOARD_DIR = Path(__file__).parent


@router.get("/", response_class=HTMLResponse)
async def dashboard_home():
    """Serve the main dashboard page."""
    html_path = DASHBOARD_DIR / "static" / "index.html"
    return HTMLResponse(content=html_path.read_text())
