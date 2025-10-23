# Base yang lebih baru & repo APT masih aktif
FROM python:3.11-slim-bookworm

WORKDIR /app

# Update + install deps OS dalam satu layer + retries
RUN apt-get -o Acquire::Retries=5 update \
&& apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
&& rm -rf /var/lib/apt/lists/*

# Copy hanya requirements dulu untuk caching layer pip
COPY requirements.txt /app/

RUN pip install -r requirements.txt

# Baru copy kode aplikasi
COPY . /app

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl --fail "http://localhost:${APP_PORT}/_stcore/health" || exit 1

ENTRYPOINT [ "sh", "-lc", \
  "exec streamlit run app.py --server.port=${APP_PORT} --server.address=${APP_HOST}" ]
