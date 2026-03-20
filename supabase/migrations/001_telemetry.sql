-- brain-mcp telemetry schema (deployed on shelet project)
-- Table: public.telemetry_events (prefixed to avoid collision)
-- RLS: anon can INSERT only, service_role can read

CREATE TABLE IF NOT EXISTS telemetry_events (
  id         bigint generated always as identity primary key,
  ts         timestamptz not null default now(),
  machine_id text not null,
  event      text not null,
  props      jsonb not null default '{}',
  version    text,
  os         text,
  python     text
);

CREATE INDEX IF NOT EXISTS idx_tel_events_event    ON telemetry_events (event);
CREATE INDEX IF NOT EXISTS idx_tel_events_ts       ON telemetry_events (ts);
CREATE INDEX IF NOT EXISTS idx_tel_events_machine  ON telemetry_events (machine_id);
CREATE INDEX IF NOT EXISTS idx_tel_events_version  ON telemetry_events (version);
CREATE INDEX IF NOT EXISTS idx_tel_events_combo    ON telemetry_events (machine_id, event);

ALTER TABLE telemetry_events ENABLE ROW LEVEL SECURITY;
CREATE POLICY allow_anon_insert ON telemetry_events FOR INSERT TO anon WITH CHECK (true);

-- Views for dashboarding

CREATE OR REPLACE VIEW telemetry_tool_popularity AS
SELECT
  props->>'tool' as tool,
  count(*) as calls,
  count(distinct machine_id) as users,
  avg((props->>'latency_ms')::float) as avg_latency_ms,
  percentile_cont(0.95) within group (order by (props->>'latency_ms')::float) as p95_latency_ms
FROM telemetry_events
WHERE event = 'tool_called' AND props->>'tool' IS NOT NULL
GROUP BY 1 ORDER BY calls DESC;

CREATE OR REPLACE VIEW telemetry_setup_funnel AS
SELECT event, count(*) as total, count(distinct machine_id) as unique_machines
FROM telemetry_events
WHERE event IN ('setup_started', 'setup_completed', 'setup_failed', 'first_query')
GROUP BY 1;

CREATE OR REPLACE VIEW telemetry_error_rates AS
SELECT
  props->>'tool' as tool, props->>'error_type' as error_type,
  count(*) as count, min(ts) as first_seen, max(ts) as last_seen
FROM telemetry_events WHERE event = 'error'
GROUP BY 1, 2 ORDER BY count DESC;

CREATE OR REPLACE VIEW telemetry_weekly_growth AS
SELECT
  date_trunc('week', ts)::date as week,
  count(distinct machine_id) as active_machines,
  count(distinct case when event = 'setup_completed' then machine_id end) as new_installs,
  count(case when event = 'tool_called' then 1 end) as total_tool_calls
FROM telemetry_events GROUP BY 1 ORDER BY 1 DESC;
