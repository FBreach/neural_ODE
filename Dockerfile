FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

WORKDIR /app

# cpu only pytorch
RUN python -m pip install --no-cache-dir \
    "torch>=2.0" --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md LICENSE.md NOTICE.md ./
COPY src/ ./src/

RUN python -m pip install --no-cache-dir ".[workflow]"

COPY Snakefile ./
COPY examples/config.yaml ./examples/config.yaml
COPY examples/scripts/ ./examples/scripts/
COPY examples/forecast/fetch_oxford_weather.py ./examples/forecast/

RUN useradd --create-home runner \
    && chown -R runner:runner /app

USER runner

CMD ["snakemake", "--cores", "1"]