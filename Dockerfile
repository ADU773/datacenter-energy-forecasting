# Serving / CLI image (CPU-only PyTorch keeps it small; GPU training is meant for a workstation).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg

WORKDIR /app

RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install .

RUN useradd --create-home --uid 1000 app && mkdir -p /work && chown app:app /work
USER app
WORKDIR /work

EXPOSE 8000

# --bootstrap generates data and trains a quick model on first start if no registry exists.
CMD ["dcforecast", "serve", "--host", "0.0.0.0", "--port", "8000", "--bootstrap"]
