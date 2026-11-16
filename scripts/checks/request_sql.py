import json
from datetime import date, datetime
from decimal import Decimal
from uuid import UUID

import psycopg
from psycopg.rows import dict_row

from ledgerlight.common import database_url, path, write_json


def scalar(value):
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (date, datetime, UUID)):
        return str(value)
    return value


results = {}
with psycopg.connect(database_url(), row_factory=dict_row) as connection:
    for file in sorted(path("sql").glob("*.sql")):
        rows = connection.execute(file.read_text()).fetchall()
        results[file.name] = [{key: scalar(value) for key, value in row.items()} for row in rows]
write_json("data/serving/sql_slices.json", results)
print(json.dumps(results, indent=2))
