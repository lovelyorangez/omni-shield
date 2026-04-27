FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# System deps
RUN apt-get update && apt-get install -y \
    python3.12 python3.12-venv python3-pip \
    nodejs npm curl git ffmpeg \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Node + Ganache
RUN npm install -g ganache

# Install ZoKrates
RUN curl -LSfs get.zokrat.es | sh
ENV PATH="/root/.zokrates/bin:${PATH}"

WORKDIR /app

# Copy backend
COPY phase1_edge_engine/ ./phase1_edge_engine/

# Install Python deps
RUN cd phase1_edge_engine && \
    python3.12 -m venv venv && \
    . venv/bin/activate && \
    pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy ZK keys if present
COPY phase1_edge_engine/zk/ ./phase1_edge_engine/zk/

# Startup script
COPY docker-entrypoint.sh .
RUN chmod +x docker-entrypoint.sh

EXPOSE 8000 7545

ENTRYPOINT ["./docker-entrypoint.sh"]
