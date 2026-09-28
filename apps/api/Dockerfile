FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md /app/
COPY aquire_preprocessing /app/aquire_preprocessing
COPY apps /app/apps
RUN pip install --no-cache-dir '.[prototype]'

ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["uvicorn", "apps.api.aquire_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
