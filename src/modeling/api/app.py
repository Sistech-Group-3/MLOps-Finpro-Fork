from __future__ import annotations

from contextlib import asynccontextmanager
import time
import pandas as pd

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware

from .. import config as C
from .schemas import (DISCLAIMER, CompareEntry, CompareResponse, HealthResponse,
                      ModelInfo, RiskResponse, RoutePoint, RouteResponse)
from .service import RiskService
from .persistence import PredictionLogStore, make_record
from .route_cache import RouteCache

ROUTE_BW_SPACE = 50.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.service = RiskService()
    except Exception as exc:
        app.state.service = None
        print(f"[startup] service unavailable: {exc}")

    app.state.store = PredictionLogStore()

    try:
        route_cache = RouteCache(bw_space=ROUTE_BW_SPACE)
        route_cache.load()
        app.state.route_cache = route_cache
        app.state.graph = route_cache.graph if route_cache.loaded else None
        app.state.df = route_cache.df if route_cache.loaded else None
    except Exception as exc:
        print(f"[startup] route_cache unavailable: {exc}")
        app.state.route_cache = None
        app.state.graph = None
        app.state.df = None

    yield


app = FastAPI(
    title="Chicago Crime Risk Score Service",
    description="Historical-pattern risk estimation (0-100) for a location & time.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


# --------------------------------------------------------------------------- #
# Dependency-style accessors (read from app.state, not module globals)
# --------------------------------------------------------------------------- #

def _require_service(request: Request) -> RiskService:
    service = request.app.state.service
    if service is None:
        raise HTTPException(503, "Models not trained yet; run `python -m modeling.train`.")
    return service


def _resolve_model(request: Request, model_arg: str) -> str:
    svc = _require_service(request)
    if model_arg in (None, "", "default"):
        return svc.default_model()
    if model_arg == "baseline":
        return "baseline"
    available = list(svc.models.keys())
    if model_arg not in available:
        raise HTTPException(400, f"unknown model '{model_arg}'. Available: {available + ['baseline']}")
    return model_arg


def _require_route(request: Request):
    G = request.app.state.graph
    if G is None:
        raise HTTPException(
            503,
            "Walk G not loaded. Download it first by running "
            "`src/notebooks/chicago_map.ipynb`, then restart the server.",
        )
    return G


def _require_crime_df(request: Request):
    df = request.app.state.df
    if df is None:
        raise HTTPException(
            503, "Crime records needed for route scoring are unavailable."
        )
    return df


@app.get("/health", response_model=HealthResponse)
def health(request: Request):
    service = _require_service(request)
    return HealthResponse(
        status="ok",
        models_loaded=len(service.models),
        active={n: e["meta"].get("version", "?") for n, e in service.models.items()},
    )


@app.get("/risk-score", response_model=RiskResponse)
def risk_score(request: Request, lat: float, lon: float, datetime: str, model: str = "default"):
    svc = _require_service(request)
    when = _parse_datetime(datetime)
    model = _resolve_model(request, model)
    t0 = time.time()
    all_scores = svc.predict_all(lat, lon, when)
    if model not in all_scores:
        raise HTTPException(400, f"model '{model}' not available")
    score, version = all_scores[model]
    latency = (time.time() - t0) * 1000.0
    level = C.level_for(score)
    request.app.state.store.append(make_record(lat, lon, when, model, version, score, level, latency))
    return RiskResponse(
        risk_score=round(score, 2),
        level=level,
        model_used=model,
        model_version=version or "-",
        last_updated=_today(),
        disclaimer=DISCLAIMER,
    )


@app.get("/risk-score/compare", response_model=CompareResponse)
def compare(request: Request, lat: float, lon: float, datetime: str):
    svc = _require_service(request)
    when = _parse_datetime(datetime)
    scores = svc.predict_all(lat, lon, when)
    entries = [
        CompareEntry(
            model_used=name,
            model_version=version or "-",
            risk_score=round(score, 2),
            level=C.level_for(score),
        )
        for name, (score, version) in scores.items()
    ]
    return CompareResponse(lat=lat, lon=lon, datetime=str(when), estimates=entries)


@app.get("/risk-score/level-buckets")
def level_buckets():
    return {
        "buckets": [
            {"low": lo, "high": hi, "label": label}
            for lo, hi, label in C.LEVEL_BUCKETS
        ],
        "justification": (
            "The echo 0-25, 25-50, 50-75, 75-100 split into four "
            "equal-width bands matching the [1,100] normalised "
            "scale used to train the predictors."
        ),
    }


@app.get("/models")
def list_models(request: Request):
    svc = _require_service(request)
    return {"models": [ModelInfo(**m).model_dump() for m in svc.list_models()]}


@app.get("/models/{model_name}/history")
def model_history(request: Request, model_name: str):
    svc = _require_service(request)
    entry = svc.registry.get("models", {}).get(model_name)
    if not entry:
        raise HTTPException(404, f"no history for model '{model_name}'")
    history = [_shape_version(v) for v in entry.get("versions", [])]
    return {
        "model_name": model_name,
        "active_version": entry.get("active_version"),
        "versions": history,
    }


def _shape_version(v: dict) -> dict:
    return {
        "version": v.get("version"),
        "saved_at": v.get("saved_at"),
        "metrics": v.get("metrics", {}),
        "improvement_vs_baseline": v.get("improvement_vs_baseline", {}),
    }


@app.get("/logs/recent")
def recent_logs(request: Request, limit: int = 50, offset: int = 0):
    limit = max(1, min(int(limit), 1000))
    store = request.app.state.store
    rows = store.recent(limit=limit, offset=max(0, int(offset)))
    return {"total": store.count(), "returned": len(rows), "logs": rows}


# --------------------------------------------------------------------------- #
# Safe route recommendation (v1 / v2)
# --------------------------------------------------------------------------- #

def _score_routes(route_cache: RouteCache, route_points, df_data, t_query):
    from src.utils.get_safest_route_v1 import safest_route
    return safest_route(
        routes=route_points,
        crime_df=df_data,
        t_query=t_query,
        bw_space=ROUTE_BW_SPACE,
        alpha=0.7,
        beta=0.3,
        debug=False,
        tree=route_cache.tree,
        freq=route_cache.freq,
        spatial_bounds=route_cache.s_bounds,
    )


@app.get("/route/v1", response_model=RouteResponse)
def route_v1(
    request: Request,
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    datetime: str,
    k: int = Query(10, ge=1, le=50),
):
    """K-shortest-paths candidates -> densify -> safest route (get_safest_route_v1)."""
    t_query = _parse_datetime(datetime)
    G = _require_route(request)
    df = _require_crime_df(request)
    route_cache = request.app.state.route_cache

    from src.utils.get_k_shortest_paths import get_k_shortest_paths
    from src.utils.get_routes_converter import routes_converter

    routes = get_k_shortest_paths(
        G, lat1, lon1, lat2, lon2, k=k, weight="length", snap=route_cache.snap
    )
    if not routes:
        raise HTTPException(404, "No walkable path between the two points.")
    converted = routes_converter(routes, G, densify_every_m=300)

    return _to_route_response(_score_routes(route_cache, converted, df, t_query))


@app.get("/route/v2", response_model=RouteResponse)
def route_v2(
    request: Request,
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    datetime: str,
    n_routes: int = Query(20, ge=2, le=100),
    penalty_factor: float = Query(1.7, ge=1.0),
):
    """Diverse candidate routes -> optimal safety-first selection (get_safest_route_v2)."""
    t_query = _parse_datetime(datetime)
    G = _require_route(request)
    df = _require_crime_df(request)
    route_cache = request.app.state.route_cache

    from src.utils.get_safest_route_v2 import generate_diverse_routes
    from src.utils.get_routes_converter import routes_converter

    routes = generate_diverse_routes(
        G,
        lat1,
        lon1,
        lat2,
        lon2,
        n_routes=n_routes,
        penalty_factor=penalty_factor,
        weight="length",
        snap=route_cache.snap,
    )
    if not routes:
        raise HTTPException(404, "No walkable path between the two points.")
    converted = routes_converter(routes, G, densify_every_m=50)

    return _to_route_response(_score_routes(route_cache, converted, df, t_query))


def _to_route_response(result: dict) -> RouteResponse:
    route = [RoutePoint(lat=pt[0], lon=pt[1]) for pt in result["safest_route"]]
    return RouteResponse(
        route=route,
        risk_score_mean=round(float(result["R_route_mean"]), 2),
        risk_score_max=round(float(result["R_route_max"]), 2),
        disclaimer=DISCLAIMER,
    )


def _parse_datetime(s: str) -> pd.Timestamp:
    try:
        return pd.Timestamp(s)
    except Exception:
        raise HTTPException(422, "datetime must be valid ISO 8601")


def _today() -> str:
    import datetime
    return datetime.date.today().isoformat()