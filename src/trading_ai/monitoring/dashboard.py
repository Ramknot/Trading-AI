"""Local-only FastAPI dashboard consuming monitoring view models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from trading_ai.core.logging import configure_logging
from trading_ai.monitoring.exceptions import (
    MonitoringConfigurationError,
    MonitoringError,
    MonitoringIntegrityError,
    MonitoringNotFoundError,
)
from trading_ai.monitoring.service import MonitoringService
from trading_ai.monitoring.source import BacktestMonitoringSource
from trading_ai.monitoring.store import SQLiteMonitoringStore


_LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


@dataclass(frozen=True, slots=True)
class DashboardSettings:
    host: str = "127.0.0.1"
    port: int = 8080

    def __post_init__(self) -> None:
        if self.host not in _LOCAL_HOSTS:
            raise MonitoringConfigurationError(
                "Lot 8 dashboard is local-only; host must be 127.0.0.1, localhost, or ::1"
            )
        if not 1 <= self.port <= 65535:
            raise MonitoringConfigurationError("dashboard port must be in [1, 65535]")


def build_monitoring_service(data_root: Path | str = Path("data_local")) -> MonitoringService:
    root = Path(data_root)
    return MonitoringService(
        BacktestMonitoringSource(root / "backtests"),
        SQLiteMonitoringStore(root / "monitoring" / "monitoring.db"),
    )


def create_dashboard_app(
    *,
    data_root: Path | str = Path("data_local"),
    service: MonitoringService | None = None,
) -> FastAPI:
    """Create the local read-only app; no trading engine is imported or mutated."""

    monitoring_service = service or build_monitoring_service(data_root)
    package_dir = Path(__file__).resolve().parent
    templates = Jinja2Templates(directory=str(package_dir / "templates"))
    app = FastAPI(
        title="Trading AI Observability",
        version="1.0",
        docs_url=None,
        redoc_url=None,
        openapi_url="/api/v1/openapi.json",
    )
    app.state.monitoring_service = monitoring_service
    app.mount(
        "/static",
        StaticFiles(directory=str(package_dir / "static")),
        name="static",
    )
    logger = configure_logging()

    @app.middleware("http")
    async def security_headers(request: Request, call_next):  # type: ignore[no-untyped-def]
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'"
        )
        return response

    @app.exception_handler(MonitoringNotFoundError)
    async def not_found_handler(request: Request, exc: MonitoringNotFoundError):
        logger.warning(
            "dashboard resource not found",
            extra={"component": "dashboard"},
        )
        return JSONResponse(status_code=404, content={"status": "ERROR", "error": str(exc)})

    @app.exception_handler(MonitoringIntegrityError)
    async def integrity_handler(request: Request, exc: MonitoringIntegrityError):
        logger.error(
            "dashboard refused an untrusted backtest export",
            extra={"component": "dashboard"},
        )
        return JSONResponse(status_code=409, content={"status": "ERROR", "error": str(exc)})

    @app.exception_handler(MonitoringError)
    async def monitoring_handler(request: Request, exc: MonitoringError):
        logger.error("dashboard monitoring error", extra={"component": "dashboard"})
        return JSONResponse(status_code=422, content={"status": "ERROR", "error": str(exc)})

    @app.exception_handler(ValueError)
    async def value_handler(request: Request, exc: ValueError):
        return JSONResponse(status_code=422, content={"status": "ERROR", "error": str(exc)})

    @app.get("/", response_class=HTMLResponse)
    def dashboard(request: Request, run_id: str | None = None):
        return templates.TemplateResponse(
            request=request,
            name="dashboard.html",
            context={
                "runs": monitoring_service.list_runs(),
                "selected_run_id": run_id or "",
            },
        )

    @app.get("/api/v1/runs")
    def runs() -> dict[str, Any]:
        return {"runs": monitoring_service.list_runs()}

    @app.get("/api/v1/snapshot")
    def snapshot(run_id: str = Query(...)) -> dict[str, Any]:
        return monitoring_service.inspect(run_id)

    def make_section_endpoint(section_name: str):
        def endpoint(run_id: str = Query(...)):
            return monitoring_service.section(run_id, section_name)

        return endpoint

    for route, section_name in (
        ("overview", "overview"),
        ("equity", "equity"),
        ("portfolio", "portfolio"),
        ("strategies", "strategies"),
        ("regimes", "regimes"),
        ("ml", "ml"),
        ("risk", "risk"),
        ("data-quality", "data_quality"),
        ("costs", "costs"),
        ("validation", "validation"),
        ("robustness", "robustness"),
        ("paper-readiness", "paper_readiness"),
        ("health", "health"),
    ):
        app.add_api_route(
            f"/api/v1/{route}",
            make_section_endpoint(section_name),
            methods=["GET"],
            name=f"api_{route}",
        )

    @app.get("/api/v1/decisions")
    def decisions(
        run_id: str = Query(...),
        component: str | None = None,
        symbol: str | None = None,
        strategy: str | None = None,
        status: str | None = None,
        reason: str | None = None,
        limit: int = 500,
    ) -> dict[str, Any]:
        return {
            "decisions": monitoring_service.decisions(
                run_id,
                component=component,
                symbol=symbol,
                strategy=strategy,
                status=status,
                reason=reason,
                limit=limit,
            )
        }

    @app.get("/api/v1/events")
    def events(
        run_id: str = Query(...),
        event_type: str | None = None,
        symbol: str | None = None,
        strategy_name: str | None = None,
        status: str | None = None,
        limit: int = 500,
    ) -> dict[str, Any]:
        return {
            "events": monitoring_service.events(
                run_id,
                event_type=event_type,
                symbol=symbol,
                strategy_name=strategy_name,
                status=status,
                limit=limit,
            )
        }

    @app.get("/api/v1/decision-trace")
    def decision_trace(
        run_id: str = Query(...), trace_id: str | None = None
    ) -> dict[str, Any]:
        return monitoring_service.decision_trace(run_id, trace_id)

    @app.get("/api/v1/system-health")
    def system_health() -> dict[str, Any]:
        return monitoring_service.health_without_run()

    return app


def serve_dashboard(
    *, data_root: Path | str = Path("data_local"), host: str = "127.0.0.1", port: int = 8080
) -> None:
    settings = DashboardSettings(host=host, port=port)
    import uvicorn

    uvicorn.run(
        create_dashboard_app(data_root=data_root),
        host=settings.host,
        port=settings.port,
        log_level="info",
        access_log=True,
    )
