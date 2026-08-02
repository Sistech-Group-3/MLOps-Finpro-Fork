"""

Run locally: 

uvicorn modeling.api.app:app --reload


Endpoints
---------
GET /health                    — liveness/readiness
GET /risk-score                — single model prediction (default best)
GET /risk-score/compare        — all models side-by-side
GET /models                    — list models, active versions, metrics
GET /health                    — liveness/readiness
GET /risk-score                — single model prediction (default best)
GET /risk-score/compare        — all models side-by-side
GET /models.                   — list models, active versions, metrics
GET /models/{model}/history    — per-version metrics history (traceability)
GET /logs/recent               — recent prediction logs (monitoring)
GET /route/v1                  — safest route (K-shortest candidate paths)
GET /route/v2                  — safest route (diverse candidate paths)

"""

from __future__ import annotations

import time

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .. import config as C
from .schemas import (DISCLAIMER, CompareEntry, CompareResponse, HealthResponse,
                      ModelInfo, RiskResponse, RoutePoint, RouteResponse)
from .service import RiskService
from .persistence import PredictionLogStore, make_record

app = FastAPI(
    title="Chicago Crime Risk Score Service",
    description="Historical-pattern risk estimation (0-100) for a location & time.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

service = None
store = None

# Route-service resources. The walk graph is an optional dependency: it is
# downloaded and saved (chicago_walk_graph.joblib) by the chicago_map notebook.
# If it is absent the /route/* endpoints return 503 while the rest of the
# service stays healthy.
graph = None
crime_df = None

# Where the walk graph lives, relative to the modeling workspace (src/).
GRAPH_FILE = C.WORKSPACE / "graph" / "chicago_walk_graph.joblib"


@app.on_event("startup")
def _startup():
    global service, store
    try:
        service = RiskService()
    except Exception as exc:  # assets not trained yet
        service = None
        print(f"[startup] service unavailable: {exc}")
    store = PredictionLogStore()
    _load_route_resources()


def _require_service():
    if service is None:
        raise HTTPException(503, "Models not trained yet; run `python -m modeling.train`.")
    return service


def _resolve_model(model_arg: str) -> str:
    svc = _require_service()
    if model_arg in (None, "", "default"):
        return svc.default_model()
    if model_arg == "baseline":
        return "baseline"
    if model_arg not in (name := list(svc.models.keys())):
        raise HTTPException(400, f"unknown model '{model_arg}'. "
                                 f"Available: {name + ['baseline']}")
    return model_arg


def _load_route_resources():
    """Load the walk graph (and crime records) or leave them as None.

    Returns None; the graph being absent is expected until the user runs the
    chicago_map notebook, so a missing file must never crash the server.
    """
    global graph, crime_df
    graph = None
    crime_df = None
    try:
        import joblib
        graph = joblib.load(GRAPH_FILE)
        print(f"[startup] walk graph loaded ({type(graph).__name__})")
    except FileNotFoundError:
        print("[startup] chicago_walk_graph.joblib not found; /route/* will "
              "return 503. Run `src/notebooks/chicago_map.ipynb` to download it.")
        return
    except Exception as exc:
        print(f"[startup] failed to load walk graph: {exc!r}")
        return

    crime_df = _load_crime_df()


def _load_crime_df():
    """Load crime records for dynamic route risk-scoring, or None if absent."""
    try:
        import pandas as pd
        events = pd.read_parquet(C.EVENTS_FILE)
        if "Crime_Score" in events.columns:
            pass
        elif "Severity_Score" in events.columns:
            events = events.rename(columns={"Severity_Score": "Crime_Score"})
        else:
            raise ValueError("no severity/score column available for routing")
        if "Date" not in events.columns and "Parsed_Date" in events.columns:
            events = events.rename(columns={"Parsed_Date": "Date"})
        if "FBI Code" not in events.columns and "Primary Type" in events.columns:
            events["FBI Code"] = events["Primary Type"].astype(str)
        events["Date"] = pd.to_datetime(events["Date"])
        cols = ["Latitude", "Longitude", "Date", "FBI Code", "Crime_Score"]
        events = events[[c for c in cols if c in events.columns]]
        events = events.drop_duplicates().dropna(
            subset=["Latitude", "Longitude", "Date"])
        print(f"[routes] loaded crime records: {len(events)}")
        return events
    except Exception as exc:
        print(f"[routes] crime records unavailable: {exc!r}")
        return None


def _require_route():
    """Return the graph or raise 503 with a polite instruction."""
    if graph is None:
        raise HTTPException(
            503,
            "Walk graph not loaded. Download it first by running "
            "`src/notebooks/chicago_map.ipynb`, then restart the server.",
        )
    if crime_df is None:
        raise HTTPException(
            503, "Crime records needed for route scoring are unavailable.")
    return graph


def _require_crime_df():
    """Return loaded crime records or raise 503 if unavailable."""
    if crime_df is None:
        raise HTTPException(
            503, "Crime records needed for route scoring are unavailable.")
    return crime_df


@app.get("/health", response_model=HealthResponse)
def health():
    _require_service()
    return HealthResponse(status="ok",
                          models_loaded=len(service.models),
                          active={n: e["meta"].get("version", "?")
                                  for n, e in service.models.items()})


@app.get("/risk-score", response_model=RiskResponse)
def risk_score(lat: float, lon: float, datetime: str, model: str = "default"):
    svc = _require_service()
    when = _parse_datetime(datetime)
    model = _resolve_model(model)
    t0 = time.time()
    all_scores = svc.predict_all(lat, lon, when)
    if model not in all_scores:
        raise HTTPException(400, f"model '{model}' not available")
    score, version = all_scores[model]
    latency = (time.time() - t0) * 1000.0
    level = C.level_for(score)
    store.append(make_record(lat, lon, when, model, version, score, level, latency))
    return RiskResponse(risk_score=round(score, 2), level=level, model_used=model,
                        model_version=version or "-",
                        last_updated=_today(), disclaimer=DISCLAIMER)


@app.get("/risk-score/compare", response_model=CompareResponse)
def compare(lat: float, lon: float, datetime: str):
    svc = _require_service()
    when = _parse_datetime(datetime)
    scores = svc.predict_all(lat, lon, when)
    entries = []
    for name, (score, version) in scores.items():
        entries.append(CompareEntry(model_used=name,
                                    model_version=version or "-",
                                    risk_score=round(score, 2),
                                    level=C.level_for(score)))
    return CompareResponse(lat=lat, lon=lon, datetime=str(when), estimates=entries)


@app.get("/risk-score/level-buckets")
def level_buckets():
    return {"buckets": [{"low": lo, "high": hi, "label": label}
                        for lo, hi, label in C.LEVEL_BUCKETS],
            "justification": ("The echo 0-25, 25-50, 50-75, 75-100 split into four "
                              "equal-width bands matching the [1,100] normalised "
                              "scale used to train the predictors.")}


@app.get("/models")
def list_models():
    svc = _require_service()
    return {"models": [ModelInfo(**m).dict() for m in svc.list_models()]}


@app.get("/models/{model_name}/history")
def model_history(model_name: str):
    svc = _require_service()
    entry = svc.registry.get("models", {}).get(model_name)
    if not entry:
        raise HTTPException(404, f"no history for model '{model_name}'")
    history = [_shape_version(v) for v in entry.get("versions", [])]
    return {"model_name": model_name, "active_version": entry.get("active_version"),
            "versions": history}


def _shape_version(v: dict) -> dict:
    return {"version": v.get("version"),
            "saved_at": v.get("saved_at"),
            "metrics": v.get("metrics", {}),
            "improvement_vs_baseline": v.get("improvement_vs_baseline", {})}


@app.get("/logs/recent")
def recent_logs(limit: int = 50, offset: int = 0):
    limit = max(1, min(int(limit), 1000))
    rows = store.recent(limit=limit, offset=max(0, int(offset)))
    return {"total": store.count(), "returned": len(rows), "logs": rows}


# --------------------------------------------------------------------------- #
# Safe route recommendation (v1 / v2)
# --------------------------------------------------------------------------- #

def _score_routes(route_points, df, t_query):
    """Run the dynamic risk-scoring / safest-route selection (get_safest_route_v1)."""
    from src.utils.get_safest_route_v1 import safest_route
    return safest_route(
        routes=route_points, crime_df=df, t_query=t_query,
        half_life_years=15, bw_space=300, alpha=0.7, beta=0.3, debug=False,
    )


@app.get("/route/v1", response_model=RouteResponse)
def route_v1(lat1: float, lon1: float, lat2: float, lon2: float,
             datetime: str, k: int = Query(10, ge=1, le=50)):
    """K-shortest-paths candidates -> densify -> safest route (get_safest_route_v1)."""
    G = _require_route()
    df = _require_crime_df()
    t_query = _parse_datetime(datetime)

    from src.utils.get_k_shortest_paths import get_k_shortest_paths
    from src.utils.get_routes_converter import routes_converter

    routes = get_k_shortest_paths(G, lat1, lon1, lat2, lon2, k=k, weight="length")
    if not routes:
        raise HTTPException(404, "No walkable path between the two points.")
    converted = routes_converter(routes, G, densify_every_m=300)

    return _to_route_response(_score_routes(converted, df, t_query))


@app.get("/route/v2", response_model=RouteResponse)
def route_v2(lat1: float, lon1: float, lat2: float, lon2: float,
             datetime: str, n_routes: int = Query(20, ge=2, le=100),
             penalty_factor: float = Query(1.7, ge=1.0)):
    """Diverse candidate routes -> optimal safety-first selection (get_safest_route_v2)."""
    G = _require_route()
    df = _require_crime_df()
    t_query = _parse_datetime(datetime)

    from src.utils.get_safest_route_v2 import generate_diverse_routes
    from src.utils.get_routes_converter import routes_converter

    routes = generate_diverse_routes(
        G, lat1, lon1, lat2, lon2, n_routes=n_routes,
        penalty_factor=penalty_factor, weight="length")
    if not routes:
        raise HTTPException(404, "No walkable path between the two points.")
    converted = routes_converter(routes, G, densify_every_m=50)

    return _to_route_response(_score_routes(converted, df, t_query))


def _to_route_response(result: dict) -> RouteResponse:
    """Build the RouteResponse from a safest_route() result dict."""
    route = [RoutePoint(lat=pt[0], lon=pt[1]) for pt in result["safest_route"]]
    return RouteResponse(
        route=route,
        risk_score_mean=round(float(result["R_route_mean"]), 2),
        risk_score_max=round(float(result["R_route_max"]), 2),
        disclaimer=DISCLAIMER,
    )


def _parse_datetime(s: str):
    try:
        import pandas as pd
        return pd.Timestamp(s)
    except Exception:
        raise HTTPException(422, "datetime must be valid ISO 8601")


def _today() -> str:
    import datetime
    return datetime.date.today().isoformat()