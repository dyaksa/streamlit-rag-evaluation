# Dockerfile
FROM python:3.11-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# paket minimal; tambahkan lib OS sesuai kebutuhan Anda (poppler/torch dll)
RUN adduser --disabled-password --gecos "" app && \
    apt-get update && apt-get install -y --no-install-recommends \
      curl build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Streamlit env
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# healthcheck bawaan Streamlit
HEALTHCHECK CMD curl --fail http://localhost:${STREAMLIT_SERVER_PORT}/_stcore/health || exit 1

USER app
EXPOSE 8501
CMD ["streamlit", "run", "main.py", "--server.address=0.0.0.0", "--server.port=8501"]
