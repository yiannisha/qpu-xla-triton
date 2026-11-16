# Optional portable/software environment; the host supplies V3D and device access.
FROM ghcr.io/astral-sh/uv:0.11.2 AS uv
FROM python:3.12.13-slim-bookworm
COPY --from=uv /uv /usr/local/bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /artifact
COPY . /artifact
RUN uv sync --locked --group dev
ENV PATH="/artifact/.venv/bin:$PATH"
CMD ["python", "scripts/artifact.py", "smoke", "--output", "/tmp/artifact-smoke"]
