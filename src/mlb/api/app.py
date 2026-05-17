"""FastAPI application — main entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from mlb.api.routes import betting, predictions, rankings, analytics
from mlb.dashboard.routes import router as dashboard_router

app = FastAPI(
    title="That's Baseball",
    description="MLB prediction, ranking, and betting system",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routes
app.include_router(predictions.router, prefix="/api/v1/predictions", tags=["predictions"])
app.include_router(rankings.router, prefix="/api/v1/rankings", tags=["rankings"])
app.include_router(betting.router, prefix="/api/v1/betting", tags=["betting"])
app.include_router(analytics.router, prefix="/api/v1/analytics", tags=["analytics"])


# Dashboard
app.include_router(dashboard_router, tags=["dashboard"])


@app.get("/api/health")
@app.get("/api/v1/health")
async def health():
    """Health check — used by Docker HEALTHCHECK and load balancers."""
    from datetime import date
    from pathlib import Path

    today = date.today().isoformat()
    has_predictions = (Path("data/predictions") / f"{today}.json").exists()

    return {
        "status": "ok",
        "service": "thats-baseball",
        "date": today,
        "predictions_ready": has_predictions,
    }
