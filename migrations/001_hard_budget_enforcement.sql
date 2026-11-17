ALTER TABLE public.requests
    ADD COLUMN IF NOT EXISTS estimated_cost_usd NUMERIC,
    ADD COLUMN IF NOT EXISTS actual_total_cost_usd NUMERIC,
    ADD COLUMN IF NOT EXISTS budget_accounted_cost_usd NUMERIC,
    ADD COLUMN IF NOT EXISTS remaining_budget_usd NUMERIC,
    ADD COLUMN IF NOT EXISTS budget_exceeded BOOLEAN,
    ADD COLUMN IF NOT EXISTS quality_verification_status TEXT,
    ADD COLUMN IF NOT EXISTS escalation_attempted BOOLEAN NOT NULL DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS escalation_reason TEXT;

ALTER TABLE public.model_calls
    ADD COLUMN IF NOT EXISTS actual_cost_usd NUMERIC,
    ADD COLUMN IF NOT EXISTS estimated_cost_usd NUMERIC,
    ADD COLUMN IF NOT EXISTS cost_is_estimated BOOLEAN NOT NULL DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS call_type TEXT NOT NULL DEFAULT 'generation',
    ADD COLUMN IF NOT EXISTS call_succeeded BOOLEAN NOT NULL DEFAULT TRUE,
    ADD COLUMN IF NOT EXISTS error_type TEXT;

UPDATE public.model_calls
SET actual_cost_usd = cost_usd
WHERE actual_cost_usd IS NULL
  AND cost_usd IS NOT NULL
  AND cost_is_estimated = FALSE;

UPDATE public.requests AS request
SET actual_total_cost_usd = totals.actual_cost,
    budget_accounted_cost_usd = totals.accounted_cost,
    remaining_budget_usd = CASE
        WHEN request.max_cost_usd IS NULL THEN NULL
        ELSE request.max_cost_usd - totals.accounted_cost
    END,
    budget_exceeded = CASE
        WHEN request.max_cost_usd IS NULL THEN FALSE
        ELSE totals.accounted_cost > request.max_cost_usd
    END
FROM (
    SELECT request_id,
           SUM(COALESCE(actual_cost_usd, 0)) AS actual_cost,
           SUM(COALESCE(cost_usd, 0)) AS accounted_cost
    FROM public.model_calls
    GROUP BY request_id
) AS totals
WHERE request.id = totals.request_id
  AND request.actual_total_cost_usd IS NULL;
