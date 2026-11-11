import psycopg
from psycopg.types.json import Jsonb

from ledgerlight.common import database_url

DDL = """
CREATE TABLE IF NOT EXISTS ask_requests (
 request_id UUID PRIMARY KEY,
 created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
 question TEXT NOT NULL,
 filters JSONB NOT NULL,
 candidates JSONB NOT NULL,
 answer TEXT NOT NULL,
 citations JSONB NOT NULL,
 latencies JSONB NOT NULL,
 model_versions JSONB NOT NULL,
 status TEXT NOT NULL,
 error_kind TEXT,
 generation JSONB NOT NULL
)
"""


def initialize():
    with psycopg.connect(database_url()) as connection:
        connection.execute(DDL)


def record(
    request_id,
    question,
    filters,
    retrieval,
    answer,
    citations,
    latencies,
    status="ok",
    error_kind=None,
    generation=None,
):
    with psycopg.connect(database_url()) as connection:
        connection.execute(
            """
          INSERT INTO ask_requests
          (request_id, question, filters, candidates, answer, citations, latencies,
           model_versions, status, error_kind, generation)
          VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """,
            (
                request_id,
                question,
                Jsonb(filters),
                Jsonb(
                    {
                        "trace": retrieval.get("trace", []),
                        "stages": retrieval.get("stage_candidates", {}),
                    }
                ),
                answer,
                Jsonb(citations),
                Jsonb(latencies),
                Jsonb(retrieval.get("versions", {})),
                status,
                error_kind,
                Jsonb(generation or {}),
            ),
        )
