SELECT error_kind, count(*) AS requests,
       percentile_cont(0.95) WITHIN GROUP (ORDER BY (latencies->>'e2e')::double precision) AS p95_seconds
FROM ask_requests
WHERE status = 'error'
GROUP BY error_kind
ORDER BY requests DESC;
