# task #4

import json
import os
import time
from datetime import datetime

LOG_DIR = os.path.dirname(__file__)

def get_log_file():
    date_str = time.strftime("%Y-%m-%d")
    return os.path.join(LOG_DIR, f"requests_log_{date_str}.jsonl")

def log_request(provider, prompt, response, start_time, end_time, cost=None, status="success"):
    log_entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "provider": provider,
        "prompt_length": len(prompt),
        "response_length": len(response) if response else 0,
        "latency_seconds": round(end_time - start_time, 3),
        "cost_usd": cost,
        "status": status
    }

    os.makedirs(os.path.dirname(LOG_DIR), exist_ok=True)
    with open(get_log_file(), "a") as f:
        json.dump(log_entry, f)
        f.write("\n")
