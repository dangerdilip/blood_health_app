import multiprocessing

# Number of worker processes
workers = max(1, multiprocessing.cpu_count() // 2)

# Use Uvicorn worker for FastAPI
worker_class = "uvicorn.workers.UvicornWorker"

# Timeouts (important for ML inference)
timeout = 120
keepalive = 5

# Logging
loglevel = "info"
accesslog = "-"
errorlog = "-"
