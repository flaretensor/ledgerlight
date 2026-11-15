FROM ghcr.io/astral-sh/uv:0.13.7 AS uv
FROM ubuntu:26.04
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates libgomp1 libnuma1 libatomic1
COPY --from=uv /uv /uvx /usr/local/bin/
WORKDIR /workspace
ENV UV_PYTHON_INSTALL_DIR=/workspace/.python UV_CACHE_DIR=/workspace/.cache/uv UV_PROJECT_ENVIRONMENT=/opt/venv
ENV PIP_CACHE_DIR=/workspace/.cache/pip XDG_CONFIG_HOME=/workspace/.config
ENV HF_HOME=/workspace/.cache/huggingface XDG_CACHE_HOME=/workspace/.cache HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_XET=1 DO_NOT_TRACK=1 STREAMLIT_BROWSER_GATHER_USAGE_STATS=false MLFLOW_DISABLE_TELEMETRY=true
ENV PATH=/opt/venv/bin:$PATH LEDGERLIGHT_ROOT=/workspace MLFLOW_TRACKING_URI=http://127.0.0.1:5000
COPY --from=python_runtime / /workspace/.python/
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=from=dependency_cache,target=/workspace/.cache/uv,rw UV_LINK_MODE=copy uv sync --offline --frozen --no-dev --no-install-project
COPY src ./src
RUN --mount=from=dependency_cache,target=/workspace/.cache/uv,rw UV_LINK_MODE=copy uv sync --offline --frozen --no-dev
COPY scripts/brand_ui.py /opt/brand_ui.py
RUN python /opt/brand_ui.py
COPY scripts ./scripts
COPY config ./config
COPY params.yaml services.lock.json ./
RUN rm -rf /home/ubuntu && rmdir /home
ENV PYTHONDONTWRITEBYTECODE=1
COPY .streamlit/config.toml ./.streamlit/config.toml
USER 1000:1000
