-- Online slick_to_aoi backfill for a new shared-dataset AOI type.
--
-- This script is intentionally operator-run, not an Alembic migration:
-- it creates private maintenance state, validates a preloaded staging table,
-- and exposes procedures that process active slicks in short transactions.
--
-- Required psql variables:
--   aoi_short_name      Stable aoi_type.short_name, e.g. 'NEW_AOI'
--   aoi_long_name       Human-facing aoi_type.long_name
--   asset_slug          Shared-datasets asset slug
--   ext_id_field        Source dataset property used as the external id
--   stage_table         Preloaded staging table, e.g. maintenance.aoi_stage_new_aoi
--
-- Optional psql variables:
--   display_name_field  Source dataset property used as AOI name
--   citation            aoi_type.citation
--   source_url          aoi_type.source_url
--   dataset_version     Exact shared-dataset resolved version used for staging
--   batch_size          Slick id window size, default 5000
--
-- Recommended usage:
--   1) Deploy the UI config guard that excludes hidden asset-backed AOIs from
--      aoiNames/aoiExternalIds, then rotate/purge the config cache.
--   2) Load the exact FGB version into the private staging table. The table must
--      have columns: ext_id text, name text, geom geometry(MultiPolygon,4326).
--      Keep it outside public. Example:
--        CREATE SCHEMA IF NOT EXISTS maintenance;
--        CREATE TABLE maintenance.aoi_stage_new_aoi (
--          ext_id text NOT NULL,
--          name text NOT NULL,
--          geom geometry(MultiPolygon, 4326) NOT NULL
--        );
--        -- Load with ogr2ogr or another controlled import path.
--   3) Run this file with the variables above. Do not wrap it in BEGIN.
--   4) Run batches:
--        CALL maintenance.run_shared_dataset_aoi_backfill('NEW_AOI');
--      Repeat or use p_max_batches for controlled windows:
--        CALL maintenance.run_shared_dataset_aoi_backfill('NEW_AOI', 25);
--   5) Run a final catch-up by rerunning this file, then running batches again.
--   6) Validate:
--        SELECT * FROM maintenance.validate_shared_dataset_aoi_backfill('NEW_AOI');
--   7) After validation passes, enable future orchestrator joins while keeping
--      the type hidden from filters:
--        CALL maintenance.finish_shared_dataset_aoi_backfill('NEW_AOI');
--      Only a human maintainer should later set filter_toggle = TRUE.

\set ON_ERROR_STOP on

\if :{?aoi_short_name}
\else
  \echo 'Required variable missing: aoi_short_name'
  \quit 1
\endif
\if :{?aoi_long_name}
\else
  \echo 'Required variable missing: aoi_long_name'
  \quit 1
\endif
\if :{?asset_slug}
\else
  \echo 'Required variable missing: asset_slug'
  \quit 1
\endif
\if :{?ext_id_field}
\else
  \echo 'Required variable missing: ext_id_field'
  \quit 1
\endif
\if :{?stage_table}
\else
  \echo 'Required variable missing: stage_table'
  \quit 1
\endif
\if :{?display_name_field}
\else
  \set display_name_field ''
\endif
\if :{?citation}
\else
  \set citation ''
\endif
\if :{?source_url}
\else
  \set source_url ''
\endif
\if :{?dataset_version}
\else
  \set dataset_version ''
\endif
\if :{?batch_size}
\else
  \set batch_size 5000
\endif

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'aoi_type'
          AND column_name IN ('filter_toggle', 'read_perm', 'access_type', 'properties')
        GROUP BY table_schema, table_name
        HAVING count(*) = 4
    ) THEN
        RAISE EXCEPTION 'Target DB is missing AOI access columns. Apply the AOI access migration/manual script before this backfill.';
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'aoi'
          AND column_name = 'ext_id'
    ) THEN
        RAISE EXCEPTION 'Target DB is missing public.aoi.ext_id. Apply the AOI access migration/manual script before this backfill.';
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'ck_aoi_type_access_properties'
          AND conrelid = 'public.aoi_type'::regclass
    ) THEN
        RAISE EXCEPTION 'Target DB is missing ck_aoi_type_access_properties. Apply the AOI access migration/manual script before this backfill.';
    END IF;
END $$;

-- Hot-table indexes. These statements must run outside an explicit transaction.
-- They are idempotent by the canonical names used in this repository.
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_slick_to_aoi_slick
    ON public.slick_to_aoi (slick);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_slick_to_aoi_aoi
    ON public.slick_to_aoi (aoi);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_aoi_type_ext_id
    ON public.aoi (type, ext_id);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_slick_geom
    ON public.slick
    USING gist ((geometry::geometry));

CREATE SCHEMA IF NOT EXISTS maintenance;

CREATE TABLE IF NOT EXISTS maintenance.shared_dataset_aoi_backfill_run (
    aoi_type_short_name text PRIMARY KEY,
    aoi_type_id bigint NOT NULL REFERENCES public.aoi_type(id),
    stage_table regclass NOT NULL,
    asset_slug text NOT NULL,
    dataset_version text,
    batch_size bigint NOT NULL CHECK (batch_size > 0),
    next_slick_id bigint NOT NULL,
    max_slick_id_at_start bigint NOT NULL,
    status text NOT NULL CHECK (status IN ('pending', 'running', 'completed')),
    total_slick_rows bigint NOT NULL DEFAULT 0,
    total_match_rows bigint NOT NULL DEFAULT 0,
    total_aoi_rows_inserted bigint NOT NULL DEFAULT 0,
    total_slick_to_aoi_rows_inserted bigint NOT NULL DEFAULT 0,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    completed_at timestamptz
);

CREATE TABLE IF NOT EXISTS maintenance.shared_dataset_aoi_backfill_batch (
    id bigserial PRIMARY KEY,
    aoi_type_short_name text NOT NULL
        REFERENCES maintenance.shared_dataset_aoi_backfill_run(aoi_type_short_name),
    lo bigint NOT NULL,
    hi bigint NOT NULL,
    slick_rows bigint NOT NULL DEFAULT 0,
    match_rows bigint NOT NULL DEFAULT 0,
    aoi_rows_inserted bigint NOT NULL DEFAULT 0,
    slick_to_aoi_rows_inserted bigint NOT NULL DEFAULT 0,
    started_at timestamptz NOT NULL DEFAULT now(),
    finished_at timestamptz,
    UNIQUE (aoi_type_short_name, lo, hi)
);

SELECT set_config('maintenance.aoi_short_name', :'aoi_short_name', false);
SELECT set_config('maintenance.aoi_long_name', :'aoi_long_name', false);
SELECT set_config('maintenance.asset_slug', :'asset_slug', false);
SELECT set_config('maintenance.ext_id_field', :'ext_id_field', false);
SELECT set_config('maintenance.display_name_field', :'display_name_field', false);
SELECT set_config('maintenance.citation', :'citation', false);
SELECT set_config('maintenance.source_url', :'source_url', false);
SELECT set_config('maintenance.dataset_version', :'dataset_version', false);
SELECT set_config('maintenance.stage_table', :'stage_table', false);
SELECT set_config('maintenance.batch_size', :'batch_size', false);

DO $$
DECLARE
    v_aoi_short_name text := current_setting('maintenance.aoi_short_name');
    v_aoi_long_name text := current_setting('maintenance.aoi_long_name');
    v_asset_slug text := current_setting('maintenance.asset_slug');
    v_ext_id_field text := current_setting('maintenance.ext_id_field');
    v_display_name_field text := current_setting('maintenance.display_name_field');
    v_citation text := current_setting('maintenance.citation');
    v_source_url text := current_setting('maintenance.source_url');
    v_dataset_version text := current_setting('maintenance.dataset_version');
    v_stage_table_text text := current_setting('maintenance.stage_table');
    v_batch_size bigint := current_setting('maintenance.batch_size')::bigint;
    v_stage_table regclass;
    v_stage_schema text;
    v_stage_relname text;
    v_stage_ident text;
    v_idx_prefix text;
    v_read_perm_id bigint;
    v_aoi_type_id bigint;
    v_min_slick_id bigint;
    v_max_slick_id bigint;
    v_count bigint;
    v_properties jsonb;
BEGIN
    SELECT to_regclass(v_stage_table_text) INTO v_stage_table;
    IF v_stage_table IS NULL THEN
        RAISE EXCEPTION 'Staging table % does not exist', v_stage_table_text;
    END IF;

    SELECT n.nspname, c.relname, format('%I.%I', n.nspname, c.relname)
    INTO v_stage_schema, v_stage_relname, v_stage_ident
    FROM pg_class c
    JOIN pg_namespace n ON n.oid = c.relnamespace
    WHERE c.oid = v_stage_table;

    IF v_stage_schema = 'public' THEN
        RAISE EXCEPTION 'Staging table must not live in public: %', v_stage_ident;
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = v_stage_schema
          AND table_name = v_stage_relname
          AND column_name IN ('ext_id', 'name', 'geom')
        GROUP BY table_schema, table_name
        HAVING count(*) = 3
    ) THEN
        RAISE EXCEPTION 'Staging table % must have ext_id, name, and geom columns', v_stage_ident;
    END IF;

    EXECUTE format('SELECT count(*) FROM %s', v_stage_ident) INTO v_count;
    IF v_count = 0 THEN
        RAISE EXCEPTION 'Staging table % is empty', v_stage_ident;
    END IF;

    EXECUTE format(
        'SELECT count(*) FROM %s WHERE NULLIF(ext_id, '''') IS NULL',
        v_stage_ident
    )
    INTO v_count;
    IF v_count > 0 THEN
        RAISE EXCEPTION 'Staging table % has % rows with empty ext_id', v_stage_ident, v_count;
    END IF;

    EXECUTE format(
        'SELECT count(*) FROM (SELECT ext_id FROM %s GROUP BY ext_id HAVING count(*) > 1) dup',
        v_stage_ident
    )
    INTO v_count;
    IF v_count > 0 THEN
        RAISE NOTICE 'Staging table % has % duplicate ext_id values; matches will be grouped by ext_id', v_stage_ident, v_count;
    END IF;

    EXECUTE format(
        $sql$
        SELECT count(*)
        FROM %s
        WHERE geom IS NULL
           OR ST_IsEmpty(geom)
           OR ST_SRID(geom) <> 4326
           OR NOT ST_IsValid(geom)
           OR GeometryType(geom) NOT IN ('POLYGON', 'MULTIPOLYGON')
        $sql$,
        v_stage_ident
    )
    INTO v_count;
    IF v_count > 0 THEN
        RAISE EXCEPTION 'Staging table % has % invalid, empty, non-4326, or non-polygon geometries', v_stage_ident, v_count;
    END IF;

    v_idx_prefix := 'idx_' || substr(md5(v_stage_ident), 1, 16);
    EXECUTE format(
        'CREATE INDEX IF NOT EXISTS %I ON %s USING gist (geom)',
        v_idx_prefix || '_geom',
        v_stage_ident
    );
    EXECUTE format(
        'CREATE INDEX IF NOT EXISTS %I ON %s (ext_id)',
        v_idx_prefix || '_ext_id',
        v_stage_ident
    );
    EXECUTE format('ANALYZE %s', v_stage_ident);

    SELECT id INTO v_read_perm_id
    FROM public.permission
    WHERE short_name = 'any'
    ORDER BY id
    LIMIT 1;

    IF v_read_perm_id IS NULL THEN
        RAISE EXCEPTION 'Expected permission short_name=any before AOI backfill';
    END IF;

    v_properties := jsonb_strip_nulls(jsonb_build_object(
        'asset_slug', v_asset_slug,
        'ext_id_field', v_ext_id_field,
        'display_name_field', NULLIF(v_display_name_field, ''),
        'dataset_version', NULLIF(v_dataset_version, '')
    ));

    INSERT INTO public.aoi_type (
        table_name,
        long_name,
        short_name,
        source_url,
        citation,
        update_time,
        filter_toggle,
        read_perm,
        access_type,
        properties
    )
    VALUES (
        NULL,
        v_aoi_long_name,
        v_aoi_short_name,
        NULLIF(v_source_url, ''),
        NULLIF(v_citation, ''),
        now(),
        FALSE,
        v_read_perm_id,
        NULL,
        v_properties
    )
    ON CONFLICT (short_name) DO UPDATE
    SET
        table_name = NULL,
        long_name = EXCLUDED.long_name,
        source_url = EXCLUDED.source_url,
        citation = EXCLUDED.citation,
        update_time = now(),
        filter_toggle = FALSE,
        read_perm = COALESCE(public.aoi_type.read_perm, EXCLUDED.read_perm),
        access_type = NULL,
        properties = EXCLUDED.properties
    RETURNING id INTO v_aoi_type_id;

    SELECT
        COALESCE(min(id), 0),
        COALESCE(max(id), -1)
    INTO v_min_slick_id, v_max_slick_id
    FROM public.slick
    WHERE active;

    INSERT INTO maintenance.shared_dataset_aoi_backfill_run (
        aoi_type_short_name,
        aoi_type_id,
        stage_table,
        asset_slug,
        dataset_version,
        batch_size,
        next_slick_id,
        max_slick_id_at_start,
        status
    )
    VALUES (
        v_aoi_short_name,
        v_aoi_type_id,
        v_stage_table,
        v_asset_slug,
        NULLIF(v_dataset_version, ''),
        v_batch_size,
        v_min_slick_id,
        v_max_slick_id,
        CASE WHEN v_min_slick_id > v_max_slick_id THEN 'completed' ELSE 'pending' END
    )
    ON CONFLICT (aoi_type_short_name) DO UPDATE
    SET
        aoi_type_id = EXCLUDED.aoi_type_id,
        stage_table = EXCLUDED.stage_table,
        asset_slug = EXCLUDED.asset_slug,
        dataset_version = EXCLUDED.dataset_version,
        batch_size = EXCLUDED.batch_size,
        max_slick_id_at_start = GREATEST(
            maintenance.shared_dataset_aoi_backfill_run.max_slick_id_at_start,
            EXCLUDED.max_slick_id_at_start
        ),
        status = CASE
            WHEN maintenance.shared_dataset_aoi_backfill_run.next_slick_id
                 <= GREATEST(
                     maintenance.shared_dataset_aoi_backfill_run.max_slick_id_at_start,
                     EXCLUDED.max_slick_id_at_start
                 )
                THEN 'pending'
            ELSE 'completed'
        END,
        completed_at = CASE
            WHEN maintenance.shared_dataset_aoi_backfill_run.next_slick_id
                 <= GREATEST(
                     maintenance.shared_dataset_aoi_backfill_run.max_slick_id_at_start,
                     EXCLUDED.max_slick_id_at_start
                 )
                THEN NULL
            ELSE maintenance.shared_dataset_aoi_backfill_run.completed_at
        END,
        updated_at = now();

    RAISE NOTICE 'Prepared shared-dataset AOI backfill for %, type id %, stage %, active slick id range [%..%]',
        v_aoi_short_name,
        v_aoi_type_id,
        v_stage_ident,
        v_min_slick_id,
        v_max_slick_id;
END $$;

CREATE OR REPLACE PROCEDURE maintenance.run_shared_dataset_aoi_backfill(
    p_aoi_type_short_name text,
    p_max_batches integer DEFAULT NULL,
    p_sleep_seconds double precision DEFAULT 0.05,
    p_lock_timeout text DEFAULT '1s',
    p_statement_timeout text DEFAULT '10min'
)
LANGUAGE plpgsql
AS $$
DECLARE
    v_aoi_type_id bigint;
    v_stage_table regclass;
    v_stage_ident text;
    v_batch_size bigint;
    v_lo bigint;
    v_hi bigint;
    v_max_slick_id bigint;
    v_slick_rows bigint;
    v_match_rows bigint;
    v_aoi_rows_inserted bigint;
    v_slick_to_aoi_rows_inserted bigint;
    v_batches_run integer := 0;
    v_lock_key bigint := hashtext('shared_dataset_aoi_backfill:' || p_aoi_type_short_name);
    v_sql text;
BEGIN
    IF NOT pg_try_advisory_lock(v_lock_key) THEN
        RAISE EXCEPTION 'Backfill is already running for AOI type %', p_aoi_type_short_name;
    END IF;

    LOOP
        IF p_max_batches IS NOT NULL AND v_batches_run >= p_max_batches THEN
            PERFORM pg_advisory_unlock(v_lock_key);
            RETURN;
        END IF;

        SELECT
            r.aoi_type_id,
            r.stage_table,
            format('%I.%I', n.nspname, c.relname),
            r.batch_size,
            r.next_slick_id,
            r.max_slick_id_at_start
        INTO
            v_aoi_type_id,
            v_stage_table,
            v_stage_ident,
            v_batch_size,
            v_lo,
            v_max_slick_id
        FROM maintenance.shared_dataset_aoi_backfill_run r
        JOIN pg_class c ON c.oid = r.stage_table
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE r.aoi_type_short_name = p_aoi_type_short_name
        FOR UPDATE OF r;

        IF NOT FOUND THEN
            PERFORM pg_advisory_unlock(v_lock_key);
            RAISE EXCEPTION 'No prepared backfill run for AOI type %', p_aoi_type_short_name;
        END IF;

        IF v_lo > v_max_slick_id THEN
            UPDATE maintenance.shared_dataset_aoi_backfill_run
            SET
                status = 'completed',
                completed_at = COALESCE(completed_at, now()),
                updated_at = now()
            WHERE aoi_type_short_name = p_aoi_type_short_name;

            COMMIT;
            PERFORM pg_advisory_unlock(v_lock_key);
            RETURN;
        END IF;

        v_hi := LEAST(v_lo + v_batch_size, v_max_slick_id + 1);

        EXECUTE format('SET LOCAL lock_timeout = %L', p_lock_timeout);
        EXECUTE format('SET LOCAL statement_timeout = %L', p_statement_timeout);

        UPDATE maintenance.shared_dataset_aoi_backfill_run
        SET status = 'running', updated_at = now()
        WHERE aoi_type_short_name = p_aoi_type_short_name;

        INSERT INTO maintenance.shared_dataset_aoi_backfill_batch (
            aoi_type_short_name,
            lo,
            hi
        )
        VALUES (p_aoi_type_short_name, v_lo, v_hi)
        ON CONFLICT (aoi_type_short_name, lo, hi) DO NOTHING;

        v_sql := format(
            $sql$
            WITH batch_slicks AS MATERIALIZED (
                SELECT id, geometry::geometry AS geom
                FROM public.slick
                WHERE active
                  AND id >= $1
                  AND id < $2
            ),
            matches AS MATERIALIZED (
                SELECT DISTINCT
                    s.id AS slick_id,
                    st.ext_id,
                    st.name
                FROM batch_slicks s
                JOIN %s st
                  ON s.geom && st.geom
                 AND ST_Intersects(s.geom, st.geom)
            ),
            matched_aoi AS MATERIALIZED (
                SELECT ext_id, MIN(name) AS name
                FROM matches
                GROUP BY ext_id
            ),
            existing_aoi AS MATERIALIZED (
                SELECT MIN(id) AS aoi_id, ext_id
                FROM public.aoi
                WHERE type = $3
                  AND ext_id IN (SELECT ext_id FROM matched_aoi)
                GROUP BY ext_id
            ),
            inserted_aoi AS (
                INSERT INTO public.aoi (type, ext_id, name)
                SELECT $3, m.ext_id, COALESCE(m.name, m.ext_id)
                FROM matched_aoi m
                LEFT JOIN existing_aoi e USING (ext_id)
                WHERE e.aoi_id IS NULL
                RETURNING id AS aoi_id, ext_id
            ),
            aoi_lookup AS MATERIALIZED (
                SELECT aoi_id, ext_id FROM existing_aoi
                UNION ALL
                SELECT aoi_id, ext_id FROM inserted_aoi
            ),
            inserted_slick_to_aoi AS (
                INSERT INTO public.slick_to_aoi (slick, aoi)
                SELECT DISTINCT m.slick_id, l.aoi_id
                FROM matches m
                JOIN aoi_lookup l USING (ext_id)
                ON CONFLICT DO NOTHING
                RETURNING 1
            )
            SELECT
                (SELECT count(*) FROM batch_slicks)::bigint,
                (SELECT count(*) FROM matches)::bigint,
                (SELECT count(*) FROM inserted_aoi)::bigint,
                (SELECT count(*) FROM inserted_slick_to_aoi)::bigint
            $sql$,
            v_stage_ident
        );

        EXECUTE v_sql
        INTO
            v_slick_rows,
            v_match_rows,
            v_aoi_rows_inserted,
            v_slick_to_aoi_rows_inserted
        USING v_lo, v_hi, v_aoi_type_id;

        UPDATE maintenance.shared_dataset_aoi_backfill_batch
        SET
            slick_rows = v_slick_rows,
            match_rows = v_match_rows,
            aoi_rows_inserted = v_aoi_rows_inserted,
            slick_to_aoi_rows_inserted = v_slick_to_aoi_rows_inserted,
            finished_at = now()
        WHERE aoi_type_short_name = p_aoi_type_short_name
          AND lo = v_lo
          AND hi = v_hi;

        UPDATE maintenance.shared_dataset_aoi_backfill_run
        SET
            next_slick_id = v_hi,
            status = CASE WHEN v_hi > v_max_slick_id THEN 'completed' ELSE 'pending' END,
            total_slick_rows = total_slick_rows + v_slick_rows,
            total_match_rows = total_match_rows + v_match_rows,
            total_aoi_rows_inserted = total_aoi_rows_inserted + v_aoi_rows_inserted,
            total_slick_to_aoi_rows_inserted =
                total_slick_to_aoi_rows_inserted + v_slick_to_aoi_rows_inserted,
            completed_at = CASE WHEN v_hi > v_max_slick_id THEN now() ELSE completed_at END,
            updated_at = now()
        WHERE aoi_type_short_name = p_aoi_type_short_name;

        RAISE NOTICE 'AOI % batch [%..%) slicks %, matches %, AOIs inserted %, links inserted %',
            p_aoi_type_short_name,
            v_lo,
            v_hi,
            v_slick_rows,
            v_match_rows,
            v_aoi_rows_inserted,
            v_slick_to_aoi_rows_inserted;

        COMMIT;

        v_batches_run := v_batches_run + 1;

        IF p_sleep_seconds > 0 THEN
            PERFORM pg_sleep(p_sleep_seconds);
        END IF;
    END LOOP;
END;
$$;

CREATE OR REPLACE FUNCTION maintenance.validate_shared_dataset_aoi_backfill(
    p_aoi_type_short_name text
)
RETURNS TABLE(check_name text, value bigint)
LANGUAGE plpgsql
AS $$
DECLARE
    v_aoi_type_id bigint;
    v_stage_ident text;
    v_sql text;
BEGIN
    SELECT
        r.aoi_type_id,
        format('%I.%I', n.nspname, c.relname)
    INTO v_aoi_type_id, v_stage_ident
    FROM maintenance.shared_dataset_aoi_backfill_run r
    JOIN pg_class c ON c.oid = r.stage_table
    JOIN pg_namespace n ON n.oid = c.relnamespace
    WHERE r.aoi_type_short_name = p_aoi_type_short_name;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'No prepared backfill run for AOI type %', p_aoi_type_short_name;
    END IF;

    RETURN QUERY
    SELECT
        'duplicate_aoi_ext_ids'::text,
        count(*)::bigint
    FROM (
        SELECT ext_id
        FROM public.aoi
        WHERE type = v_aoi_type_id
        GROUP BY ext_id
        HAVING count(*) > 1
    ) dup;

    v_sql := format(
        $sql$
        WITH expected AS MATERIALIZED (
            SELECT DISTINCT
                s.id AS slick_id,
                st.ext_id
            FROM public.slick s
            JOIN %s st
              ON s.geometry::geometry && st.geom
             AND ST_Intersects(s.geometry::geometry, st.geom)
            WHERE s.active
        ),
        expected_aoi AS MATERIALIZED (
            SELECT e.slick_id, MIN(a.id) AS aoi_id
            FROM expected e
            JOIN public.aoi a
              ON a.type = $1
             AND a.ext_id = e.ext_id
            GROUP BY e.slick_id, e.ext_id
        )
        SELECT
            'missing_slick_to_aoi_rows'::text,
            count(*)::bigint
        FROM expected_aoi e
        LEFT JOIN public.slick_to_aoi sta
          ON sta.slick = e.slick_id
         AND sta.aoi = e.aoi_id
        WHERE sta.slick IS NULL
        $sql$,
        v_stage_ident
    );

    RETURN QUERY EXECUTE v_sql USING v_aoi_type_id;
END;
$$;

CREATE OR REPLACE PROCEDURE maintenance.finish_shared_dataset_aoi_backfill(
    p_aoi_type_short_name text
)
LANGUAGE plpgsql
AS $$
DECLARE
    v_run_status text;
    v_aoi_type_id bigint;
    v_duplicate_count bigint;
BEGIN
    SELECT status, aoi_type_id
    INTO v_run_status, v_aoi_type_id
    FROM maintenance.shared_dataset_aoi_backfill_run
    WHERE aoi_type_short_name = p_aoi_type_short_name;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'No prepared backfill run for AOI type %', p_aoi_type_short_name;
    END IF;

    IF v_run_status <> 'completed' THEN
        RAISE EXCEPTION 'Backfill for AOI type % is %, not completed', p_aoi_type_short_name, v_run_status;
    END IF;

    SELECT count(*)
    INTO v_duplicate_count
    FROM (
        SELECT ext_id
        FROM public.aoi
        WHERE type = v_aoi_type_id
        GROUP BY ext_id
        HAVING count(*) > 1
    ) dup;

    IF v_duplicate_count > 0 THEN
        RAISE EXCEPTION 'AOI type % has % duplicate ext_id values in public.aoi', p_aoi_type_short_name, v_duplicate_count;
    END IF;

    UPDATE public.aoi_type
    SET
        access_type = 'SHARED_DATASET',
        filter_toggle = FALSE,
        update_time = now()
    WHERE id = v_aoi_type_id;

    RAISE NOTICE 'AOI type % is now available to future orchestrator joins with filter_toggle still FALSE',
        p_aoi_type_short_name;
END;
$$;

SELECT *
FROM maintenance.shared_dataset_aoi_backfill_run
WHERE aoi_type_short_name = :'aoi_short_name';
