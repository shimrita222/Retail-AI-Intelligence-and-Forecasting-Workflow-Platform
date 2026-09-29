-- Phase 4A Step 1 — base schema, profiles/role model, and RLS foundation.
--
-- Scope: local-only foundation. This migration is designed to be applied
-- with `supabase db reset` / `supabase migration up` against a LOCAL
-- Supabase Postgres instance. It has not been pushed to any remote/staging
-- project.
--
-- Preserves the application's canonical run_id (see
-- src/services/run_registry.py: ^\d{8}T\d{6}Z-[0-9a-f]{8}$) as a UNIQUE
-- column rather than replacing it with a UUID-only model -- workflow_runs
-- keeps a surrogate UUID primary key (for FK ergonomics) alongside the real
-- app-generated run_id.
--
-- Status vocabulary mirrors RetailFlowState.status exactly
-- (src/flows/retail_flow.py) -- do not add/remove values here without
-- updating that Python contract in the same change.

-- ---------------------------------------------------------------------------
-- profiles
-- ---------------------------------------------------------------------------

create table public.profiles (
    id uuid primary key references auth.users (id) on delete cascade,
    display_name text not null,
    role text not null default 'Business'
        check (role in ('Admin', 'Analyst', 'Scientist', 'Business')),
    created_at timestamptz not null default now()
);

comment on table public.profiles is
    'One row per authenticated user. role is NOT user-editable via RLS -- '
    'see policies below. Admin role assignment is a privileged backend-only '
    'operation (service-role, bypasses RLS), never a client-writable field.';

-- New auth.users rows always provision a least-privileged Business profile.
-- Elevating a role happens later, out-of-band, via a privileged backend
-- operation -- never automatically at signup.
create function public.handle_new_user()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
begin
    insert into public.profiles (id, display_name, role)
    values (
        new.id,
        coalesce(new.raw_user_meta_data ->> 'display_name', new.email, new.id::text),
        'Business'
    );
    return new;
end;
$$;

create trigger on_auth_user_created
    after insert on auth.users
    for each row execute function public.handle_new_user();

-- Helper used by RLS policies below to check the caller's role without
-- causing RLS recursion on `profiles` itself (SECURITY DEFINER + a fixed
-- search_path is the standard Supabase pattern for this).
create function public.is_admin()
returns boolean
language sql
stable
security definer
set search_path = public
as $$
    select exists (
        select 1 from public.profiles
        where id = auth.uid() and role = 'Admin'
    );
$$;

create function public.current_profile_role()
returns text
language sql
stable
security definer
set search_path = public
as $$
    select role from public.profiles where id = auth.uid();
$$;

-- ---------------------------------------------------------------------------
-- workflow_runs
-- ---------------------------------------------------------------------------

create table public.workflow_runs (
    id uuid primary key default gen_random_uuid(),
    run_id text not null unique
        check (run_id ~ '^\d{8}T\d{6}Z-[0-9a-f]{8}$'),
    owner_id uuid references public.profiles (id) on delete set null,
    status text not null
        check (status in (
            'INITIATED', 'ANALYST_COMPLETE', 'VALIDATED', 'SCIENTIST_COMPLETE',
            'COMPLETED', 'FAILED', 'GATE_FAILED', 'VIABILITY_FAILED', 'ARTIFACT_FAILED'
        )),
    contract_status text,
    is_demo boolean not null default false,
    source_environment text not null default 'local'
        check (source_environment in ('local', 'staging', 'production')),
    dataset_scope text,
    train_row_count integer check (train_row_count is null or train_row_count >= 0),
    test_row_count integer check (test_row_count is null or test_row_count >= 0),
    started_at timestamptz,
    finished_at timestamptz,
    created_at timestamptz not null default now()
);

comment on column public.workflow_runs.run_id is
    'Canonical app-level run identifier, matches src/services/run_registry.py '
    'REAL_RUN_ID_PATTERN exactly. Never replace with the surrogate uuid id.';
comment on column public.workflow_runs.is_demo is
    'Demo/full-dataset provenance flag. No row may combine is_demo = false '
    'with source_environment = ''production'' until Phase 3B (full-dataset '
    'integration verification) actually passes -- enforced procedurally by '
    'the persistence layer for now, not yet by a DB constraint.';
comment on column public.workflow_runs.owner_id is
    'Audit/provenance only -- NOT used as an access-control filter. MVP '
    'access model is shared role-based analytics access (see RLS below).';

create index workflow_runs_owner_id_idx on public.workflow_runs (owner_id);

-- ---------------------------------------------------------------------------
-- candidate_outcomes
-- ---------------------------------------------------------------------------

create table public.candidate_outcomes (
    id uuid primary key default gen_random_uuid(),
    run_id uuid not null references public.workflow_runs (id) on delete cascade,
    model_id text not null,
    family text,
    mae numeric,
    rmse numeric,
    r2 numeric,
    usable boolean not null,
    reason_code text,
    reason_params jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    unique (run_id, model_id)
);

create index candidate_outcomes_run_id_idx on public.candidate_outcomes (run_id);

-- ---------------------------------------------------------------------------
-- selected_models  (1:1 with workflow_runs)
-- ---------------------------------------------------------------------------

create table public.selected_models (
    run_id uuid primary key references public.workflow_runs (id) on delete cascade,
    model_id text not null,
    artifact_path text not null,
    artifact_storage text not null
        check (artifact_storage in ('local', 'supabase_storage')),
    tie_break_applied boolean not null default false,
    tie_break_reason_params jsonb,
    created_at timestamptz not null default now()
);

comment on table public.selected_models is
    'Never stores the model binary itself -- artifact_path is a reference '
    'into filesystem/artifact-root (or Supabase Storage for small/medium '
    'files only). selected_model.joblib and other GB-scale binaries stay '
    'off Postgres entirely.';

-- ---------------------------------------------------------------------------
-- artifacts  (generic artifact metadata/reference table)
-- ---------------------------------------------------------------------------

create table public.artifacts (
    id uuid primary key default gen_random_uuid(),
    run_id uuid not null references public.workflow_runs (id) on delete cascade,
    artifact_type text not null
        check (artifact_type in (
            'run_metadata.json', 'evaluation_report.json', 'evaluation_report.md',
            'model_card.md', 'selected_model.joblib', 'clean_data.csv',
            'dataset_contract.json', 'eda_report.html', 'insights.md',
            'validation_result.json'
        )),
    storage_backend text not null
        check (storage_backend in ('local', 'supabase_storage')),
    path_or_key text not null,
    content_type text,
    size_bytes bigint check (size_bytes is null or size_bytes >= 0),
    sha256 text,
    created_at timestamptz not null default now(),
    unique (run_id, artifact_type)
);

create index artifacts_run_id_idx on public.artifacts (run_id);

-- ---------------------------------------------------------------------------
-- business_insights
-- ---------------------------------------------------------------------------

create table public.business_insights (
    id uuid primary key default gen_random_uuid(),
    run_id uuid not null references public.workflow_runs (id) on delete cascade,
    reason_code text not null,
    reason_params jsonb not null default '{}'::jsonb,
    kind text,
    created_at timestamptz not null default now()
);

comment on table public.business_insights is
    'Stores only language-neutral structured evidence (reason_code / '
    'reason_params). Rendered en/he narration is NOT persisted here -- it is '
    'generated at presentation time by the Phase 4C i18n layer from this '
    'evidence, so numeric/structured results remain the sole source of '
    'truth and narration can never drift from them.';

create index business_insights_run_id_idx on public.business_insights (run_id);

-- ---------------------------------------------------------------------------
-- external_data_sources  (metadata only -- no credentials, ever)
-- ---------------------------------------------------------------------------

create table public.external_data_sources (
    id uuid primary key default gen_random_uuid(),
    name text not null,
    kind text,
    schema_ref jsonb,
    owner_id uuid references public.profiles (id) on delete set null,
    is_active boolean not null default true,
    created_at timestamptz not null default now()
);

comment on table public.external_data_sources is
    'Metadata only. Connection strings/credentials must never be stored '
    'here or anywhere reachable by a CrewAI agent -- see Agent Security '
    'Boundary in CLAUDE.md and the Phase 4A specification.';

create index external_data_sources_owner_id_idx on public.external_data_sources (owner_id);

-- ---------------------------------------------------------------------------
-- Row Level Security
-- ---------------------------------------------------------------------------

alter table public.profiles enable row level security;
alter table public.workflow_runs enable row level security;
alter table public.candidate_outcomes enable row level security;
alter table public.selected_models enable row level security;
alter table public.artifacts enable row level security;
alter table public.business_insights enable row level security;
alter table public.external_data_sources enable row level security;

-- profiles: everyone can read their own row; Admins can read every profile
-- (needed to administer roles later). No INSERT/UPDATE policy exists for
-- any client role -- provisioning happens only via the trigger above
-- (SECURITY DEFINER, runs as table owner) and role changes happen only via
-- a privileged backend path that uses the service-role key and therefore
-- bypasses RLS entirely. This is a deliberate MVP-safe default: role is
-- never client-writable.
create policy profiles_select_own_or_admin on public.profiles
    for select to authenticated
    using (id = auth.uid() or public.is_admin());

-- workflow_runs: shared role-based analytics read access (approved access
-- model -- owner_id is audit-only, not a read filter). Only Admins may
-- create/update/delete runs, matching config.yaml's run_workflow permission
-- (Admin-only today); writes carry the initiating Admin's own JWT rather
-- than a privileged service-role credential (approved decision).
create policy workflow_runs_select_shared on public.workflow_runs
    for select to authenticated
    using (true);

create policy workflow_runs_insert_admin_owned on public.workflow_runs
    for insert to authenticated
    with check (public.is_admin() and owner_id = auth.uid());

create policy workflow_runs_update_admin_owned on public.workflow_runs
    for update to authenticated
    using (public.is_admin() and owner_id = auth.uid())
    with check (public.is_admin());

create policy workflow_runs_delete_admin on public.workflow_runs
    for delete to authenticated
    using (public.is_admin());

-- candidate_outcomes / selected_models / artifacts / business_insights:
-- shared read access for every authenticated role; writes restricted to the
-- Admin who owns the parent run (mirrors workflow_runs write policy so the
-- same JWT that created the run can write its children).
create policy candidate_outcomes_select_shared on public.candidate_outcomes
    for select to authenticated
    using (true);

create policy candidate_outcomes_write_admin_owned on public.candidate_outcomes
    for insert to authenticated
    with check (
        public.is_admin()
        and exists (
            select 1 from public.workflow_runs wr
            where wr.id = candidate_outcomes.run_id and wr.owner_id = auth.uid()
        )
    );

create policy selected_models_select_shared on public.selected_models
    for select to authenticated
    using (true);

create policy selected_models_write_admin_owned on public.selected_models
    for insert to authenticated
    with check (
        public.is_admin()
        and exists (
            select 1 from public.workflow_runs wr
            where wr.id = selected_models.run_id and wr.owner_id = auth.uid()
        )
    );

create policy artifacts_select_shared on public.artifacts
    for select to authenticated
    using (true);

create policy artifacts_write_admin_owned on public.artifacts
    for insert to authenticated
    with check (
        public.is_admin()
        and exists (
            select 1 from public.workflow_runs wr
            where wr.id = artifacts.run_id and wr.owner_id = auth.uid()
        )
    );

create policy business_insights_select_shared on public.business_insights
    for select to authenticated
    using (true);

create policy business_insights_write_admin_owned on public.business_insights
    for insert to authenticated
    with check (
        public.is_admin()
        and exists (
            select 1 from public.workflow_runs wr
            where wr.id = business_insights.run_id and wr.owner_id = auth.uid()
        )
    );

-- external_data_sources: shared read; Admin-only, owner-tagged writes.
create policy external_data_sources_select_shared on public.external_data_sources
    for select to authenticated
    using (true);

create policy external_data_sources_write_admin_owned on public.external_data_sources
    for insert to authenticated
    with check (public.is_admin() and owner_id = auth.uid());

create policy external_data_sources_update_admin_owned on public.external_data_sources
    for update to authenticated
    using (public.is_admin())
    with check (public.is_admin());

-- No policies are defined for the `anon` role on any table above --
-- Postgres RLS defaults to deny-all in the absence of a matching policy,
-- so anonymous callers get zero business-data access by construction.
