"""
RQ worker + queue definitions for OMNI-SHIELD async jobs.

Start a worker:
    cd phase1_edge_engine
    source venv/bin/activate
    rq worker image audio --url redis://localhost:6379

Queues:
    image  — VLM image pipeline  (heavy, GPU)
    audio  — Whisper + text model (heavy, CPU/GPU)
"""

import os

from redis import Redis
from rq import Queue

redis_conn = Redis(
    host=os.environ.get("REDIS_HOST", "localhost"),
    port=int(os.environ.get("REDIS_PORT", "6379")),
)

image_queue = Queue("image", connection=redis_conn, default_timeout=180)
audio_queue = Queue("audio", connection=redis_conn, default_timeout=300)
