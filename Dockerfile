# syntax=docker/dockerfile:1.7

FROM ghcr.io/astral-sh/uv:python3.12-bookworm AS api

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_LINK_MODE=copy

COPY pyproject.toml uv.lock uv.toml .python-version ./
RUN uv sync --frozen --no-dev

COPY src ./src
COPY data/universe ./data/universe
COPY .env.example ./.env.example

EXPOSE 8000
CMD ["uv", "run", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM oven/bun:1.3.14 AS dashboard-builder

WORKDIR /app/dashboard
COPY dashboard/package.json dashboard/bun.lock ./
RUN bun install --frozen-lockfile
COPY dashboard ./
ARG VITE_API_URL=http://localhost:8000
ARG VITE_USE_MOCKS=false
ENV VITE_API_URL=$VITE_API_URL
ENV VITE_USE_MOCKS=$VITE_USE_MOCKS
RUN bun run build

FROM nginx:1.27-alpine AS dashboard

COPY dashboard/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=dashboard-builder /app/dashboard/dist /usr/share/nginx/html
EXPOSE 80
