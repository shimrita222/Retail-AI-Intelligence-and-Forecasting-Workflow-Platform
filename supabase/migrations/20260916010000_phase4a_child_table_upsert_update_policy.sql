-- Phase 4A Step 1D correction -- UPDATE support for the three child tables
-- whose repository methods use PostgREST/Postgres upsert semantics.
--
-- Live STAGING validation (Phase 4A Step 1D, Live Run #2) found that
-- SupabaseRunRepository.save_candidate_outcomes() / save_selected_model() /
-- save_artifact_metadata() issue `INSERT ... ON CONFLICT DO UPDATE` (a
-- PostgREST `.upsert(...)` call) against candidate_outcomes, selected_models,
-- and artifacts -- exactly matching the idempotent-upsert contract already
-- documented in src/persistence/interfaces.py and already exercised by the
-- local fake-client tests (test_save_candidate_outcomes_upserts,
-- test_save_selected_model_upserts, test_save_artifact_metadata_upserts).
--
-- PostgreSQL checks privileges for every clause of a statement at plan time,
-- so `ON CONFLICT DO UPDATE` requires UPDATE privilege on the target table
-- even when no conflict actually occurs. 20260914205942_phase4a_data_api_
-- grants.sql only granted SELECT/INSERT for these three tables (its own
-- comment assumed "no update/delete policy exists, so no update/delete grant
-- is needed" -- true only for business_insights, which uses plain INSERT,
-- not upsert), so every upsert call against these three tables failed with
-- Postgres permission-denied (42501) before RLS was ever evaluated.
--
-- This migration closes that gap for exactly these three tables, and no
-- others: business_insights (plain INSERT, no upsert), workflow_runs,
-- profiles, and external_data_sources are all untouched. The anonymous
-- (public, unauthenticated) role receives no new privilege whatsoever from
-- this migration, no DELETE privilege is added anywhere, and no
-- previously-applied migration is modified.
--
-- The new UPDATE policies mirror each table's existing Admin-owned INSERT
-- policy exactly: to update a row, the caller must be Admin AND the row's
-- run_id must reference a workflow_runs row owned (owner_id = auth.uid()) by
-- that same caller. Both USING and WITH CHECK repeat this exact ownership
-- check -- USING restricts which existing rows may be touched, and WITH
-- CHECK restricts what the row may look like afterward, so an Admin cannot
-- use UPDATE to reassign a row's run_id into a workflow_run they do not own.

grant update on public.candidate_outcomes to authenticated;
grant update on public.selected_models to authenticated;
grant update on public.artifacts to authenticated;

create policy candidate_outcomes_update_admin_owned on public.candidate_outcomes
    for update to authenticated
    using (
        public.is_admin()
        and exists (
            select 1 from public.workflow_runs wr
            where wr.id = candidate_outcomes.run_id and wr.owner_id = auth.uid()
        )
    )
    with check (
        public.is_admin()
        and exists (
            select 1 from public.workflow_runs wr
            where wr.id = candidate_outcomes.run_id and wr.owner_id = auth.uid()
        )
    );

create policy selected_models_update_admin_owned on public.selected_models
    for update to authenticated
    using (
        public.is_admin()
        and exists (
            select 1 from public.workflow_runs wr
            where wr.id = selected_models.run_id and wr.owner_id = auth.uid()
        )
    )
    with check (
        public.is_admin()
        and exists (
            select 1 from public.workflow_runs wr
            where wr.id = selected_models.run_id and wr.owner_id = auth.uid()
        )
    );

create policy artifacts_update_admin_owned on public.artifacts
    for update to authenticated
    using (
        public.is_admin()
        and exists (
            select 1 from public.workflow_runs wr
            where wr.id = artifacts.run_id and wr.owner_id = auth.uid()
        )
    )
    with check (
        public.is_admin()
        and exists (
            select 1 from public.workflow_runs wr
            where wr.id = artifacts.run_id and wr.owner_id = auth.uid()
        )
    );
