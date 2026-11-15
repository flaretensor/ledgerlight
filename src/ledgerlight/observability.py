from prometheus_client import Counter, Gauge, Histogram

REQUESTS = Counter("ledgerlight_requests_total", "Requests", ["service", "status"])
ERRORS = Counter("ledgerlight_errors_total", "Errors", ["service", "kind"])
ABSTENTIONS = Counter("ledgerlight_abstentions_total", "Abstentions")
OUTCOMES = Counter("ledgerlight_ask_outcomes_total", "Questions by outcome", ["outcome"])
QUEUE = Gauge("ledgerlight_queue_depth", "Questions waiting for the single generation slot")
FIRST_TEXT = Histogram(
    "ledgerlight_first_text_seconds",
    "Queue-inclusive first answer text latency",
    buckets=(1, 2, 5, 10, 20, 30, 60, 120, 180, 300),
)
STAGES = Histogram(
    "ledgerlight_stage_seconds",
    "Wall time including queue wait",
    ["service", "stage"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30, 60, 120),
)
