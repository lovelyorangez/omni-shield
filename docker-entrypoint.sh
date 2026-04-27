#!/bin/bash
set -e

echo "Starting Omni-Shield..."

# Start Ganache in background
ganache --port 7545 --deterministic \
  --db /app/ganache-db --host 0.0.0.0 &
echo "Ganache started on port 7545"

# Wait for Ganache to be ready
sleep 3

# Start backend
cd /app/phase1_edge_engine
source venv/bin/activate
echo "Starting backend on port 8000..."
uvicorn backend_server:app --host 0.0.0.0 --port 8000
