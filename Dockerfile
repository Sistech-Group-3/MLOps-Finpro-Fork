# Dockerfile for Chicago Crime Risk Score Service
# Bundles all model artifacts and data into the image (MVP approach)

FROM python:3.12-slim

WORKDIR /app

# System deps: needed by osmnx (GDAL/GEOS), scipy, and Rtree spatial index
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgdal-dev \
    libspatialindex-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast pip resolver/installer)
RUN pip install --no-cache-dir uv

# --- Dependency layer (cached unless pyproject.toml / uv.lock changes) ---
COPY pyproject.toml uv.lock ./

# Install all production dependencies (no dev extras)
RUN uv pip install --system --no-cache \
    fastapi \
    uvicorn[standard] \
    pandas \
    numpy \
    scikit-learn \
    xgboost \
    joblib \
    httpx \
    pyarrow

# --- Application source + all bundled artifacts ---
COPY . .

# Railway automatically sets $PORT; fall back to 8000 for local Docker testing
ENV PORT=8000

# Memory Optimizations for Free Tier (500MB RAM limit)
ENV MALLOC_ARENA_MAX=2
ENV PYTHONUNBUFFERED=1
ENV WEB_CONCURRENCY=1

# Expose the port (informational only — Railway reads $PORT from env)
EXPOSE $PORT

# Start the API — $PORT is expanded at runtime by the shell
CMD uvicorn src.modeling.api.app:app --host 0.0.0.0 --port $PORT --workers 1
