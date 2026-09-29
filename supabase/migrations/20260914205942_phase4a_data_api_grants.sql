-- Phase 4A Step 1C correction — Data API grants for the `authenticated` role.
--
-- Live STAGING validation (Phase 4A Step 1C) found that
-- 20260903005542_phase4a_base_schema.sql enables RLS and defines policies
-- for `authenticated` on every table, but grants no table-level SELECT /
-- INSERT / UPDATE / DELETE privilege to `anon` or `authenticated` -- current
-- Supabase projects do not auto-expose newly created public-schema tables to
-- the Data API roles. Without these GRANTs, every authenticated request was
-- failing with a Postgres permission-denied error rather than being
-- evaluated by RLS, which defeats the shared role-based analytics access
-- model described in the base migration's own comments.
--
-- This migration only grants the minimum privilege each table already has a
-- matching RLS policy for (see base migration). It grants nothing to `anon`
-- -- the deny-by-default posture for anonymous callers is intentional and
-- unchanged (no policies exist for `anon` on any table).

-- profiles: read-only via RLS (own row or Admin); rows are provisioned only
-- by the SECURITY DEFINER trigger, which runs as table owner and needs no
-- grant to `authenticated`.
grant select on public.profiles to authenticated;

-- workflow_runs: shared read; Admin-owned insert/update/delete (RLS policies
-- restrict which rows/columns actually succeed).
grant select, insert, update, delete on public.workflow_runs to authenticated;

-- candidate_outcomes / selected_models / artifacts / business_insights:
-- shared read; Admin-owned insert only (no update/delete policy exists, so
-- no update/delete grant is needed -- children are removed only via the
-- parent workflow_runs cascade).
grant select, insert on public.candidate_outcomes to authenticated;
grant select, insert on public.selected_models to authenticated;
grant select, insert on public.artifacts to authenticated;
grant select, insert on public.business_insights to authenticated;

-- external_data_sources: shared read; Admin-owned insert/update (no delete
-- policy exists).
grant select, insert, update on public.external_data_sources to authenticated;
