SELECT model_versions->>'backend' AS backend, stage.key AS stage, count(*) AS requests,
       percentile_cont(0.50) WITHIN GROUP (ORDER BY stage.value::double precision) AS p50_seconds,
       percentile_cont(0.95) WITHIN GROUP (ORDER BY stage.value::double precision) AS p95_seconds
FROM ask_requests CROSS JOIN LATERAL jsonb_each_text(latencies) AS stage
GROUP BY model_versions->>'backend', stage.key
ORDER BY p95_seconds DESC;
