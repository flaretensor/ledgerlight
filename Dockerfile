FROM ghcr.io/astral-sh/uv:0.13.7 AS uv
FROM ubuntu:26.04
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates libgomp1 libnuma1 libatomic1
COPY --from=uv /uv /uvx /usr/local/bin/
WORKDIR /workspace
ENV UV_PYTHON_INSTALL_DIR=/workspace/.python UV_CACHE_DIR=/workspace/.cache/uv UV_PROJECT_ENVIRONMENT=/opt/venv
ENV HF_HOME=/workspace/.cache/huggingface XDG_CACHE_HOME=/workspace/.cache HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_XET=1 DO_NOT_TRACK=1 STREAMLIT_BROWSER_GATHER_USAGE_STATS=false MLFLOW_DISABLE_TELEMETRY=true
ENV PATH=/opt/venv/bin:$PATH LEDGERLIGHT_ROOT=/workspace MLFLOW_TRACKING_URI=http://127.0.0.1:5000
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev
USER 1000:1000
