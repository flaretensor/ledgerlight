SELECT request_id, question, filters, model_versions, status
FROM ask_requests
WHERE jsonb_array_length(coalesce(candidates->'trace', '[]'::jsonb)) = 0
ORDER BY created_at DESC;
