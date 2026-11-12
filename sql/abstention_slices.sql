SELECT coalesce(filters->>'ticker', 'Any') AS ticker,
       model_versions->>'method' AS method,
       model_versions->>'backend' AS backend,
       count(*) AS requests,
       avg((jsonb_array_length(citations) = 0)::int) AS abstention_rate,
       percentile_cont(0.50) WITHIN GROUP (ORDER BY (latencies->>'e2e')::double precision) AS p50_seconds,
       percentile_cont(0.95) WITHIN GROUP (ORDER BY (latencies->>'e2e')::double precision) AS p95_seconds
FROM ask_requests
GROUP BY 1, 2, 3
ORDER BY requests DESC;
